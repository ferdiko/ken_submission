import torch




class BertQueue2:
    """
    This is an adapted version of BertQueue that supports setting a maximum batch size.
    TODO: Make a queue interface that the specific queues inherit from (size, add, get).
     Then refactor and put queue implementations in workloads folder.
    """

    def __init__(self, max_bs=1000, tensor_shape=(55,), num_queues=10):
        """
        The queue manages an allocated memory region on the GPU to queue (input) samples for inference.
        :param capacity:
        :param tensor_shape:
        :param num_queues: Number of memory regions. We need to switch between them because reading for inference &
          writing can happen in parallel.
        """

        # Initialize data strcutures that hold samples.
        self.sample_ids = []

        queue_shape = (max_bs,) + tensor_shape
        self.x_queue = []
        self.x_att_queue = []
        for i in range(num_queues):
            self.x_queue.append(torch.empty(queue_shape, dtype=torch.int32, device='cuda'))
            self.x_att_queue.append(torch.empty(queue_shape, dtype=torch.int32, device='cuda'))

        # Queue state.
        self.capacity = max_bs*num_queues
        self.queue_len = 0
        self.cur_queue = 0
        self.num_queues = num_queues
        self.max_bs = max_bs

        # print(f"queue len: {self.queue_len}")
        # for i, q in enumerate(self.x_queue):
        #     print(f"{i}: {q}")



    def add(self, sample_ids, samples):
        self.sample_ids += sample_ids

        num_samples = len(sample_ids)
        x, x_att = samples
        copy_ctr = 0
        while num_samples > 0:
            idx = self.queue_len % self.max_bs
            queue_idx = int(self.queue_len / self.max_bs)
            samples_added = min(self.max_bs - idx, num_samples)

            self.x_queue[queue_idx][idx:idx+samples_added].copy_(x[copy_ctr:copy_ctr+samples_added], non_blocking=True)
            self.x_att_queue[queue_idx][idx:idx+samples_added].copy_(x[copy_ctr:copy_ctr+samples_added], non_blocking=True)

            copy_ctr += samples_added
            num_samples -= samples_added

        self.queue_len += copy_ctr
        # print(f"queue len: {self.queue_len}")
        # for i, q in enumerate(self.x_queue):
        #     print(f"{i}: {q}")

        assert num_samples == 0


    def get(self, num_samples=-1):
        """
        NOTE: For now this queue can only get all samples, it's not a full FIFO queue.
        This is problematic if the queue becomes so full that the resulting large batches
        leads to OOM. But if that happens, you're screwed anyways (huge latency), which is
        why we leave it like this for now.
        :param num_samples: Only there for compatible API.
        :return: (x tensor, x_att tensor), sample ids list
        """
        # Get tensors to return
        ret_samples = min(self.max_bs, self.queue_len)
        self.queue_len -= ret_samples
        x_cur = self.x_queue[0]
        x_att_cur = self.x_att_queue[0]
        ret_sample_ids = self.sample_ids[:ret_samples]

        # Append tensors to queue.
        self.x_queue = self.x_queue[1:] + [x_cur]
        self.x_att_queue = self.x_att_queue[1:] + [x_att_cur]
        self.sample_ids = self.sample_ids[ret_samples:]

        # print(f"queue len: {self.queue_len}")
        # for i, q in enumerate(self.x_queue):
        #     print(f"{i}: {q}")

        return (x_cur[:ret_samples], x_att_cur[:ret_samples]), ret_sample_ids


    def size(self):
        return self.queue_len




class BertQueue:
    """
    TODO: Make a queue interface that the specific queues inherit from (size, add, get).
     Then refactor and put queue implementations in workloads folder.
    """

    def __init__(self, capacity=1600, tensor_shape=(55,), num_queues=2):
        """
        The queue manages an allocated memory region on the GPU to queue (input) samples for inference.
        :param capacity:
        :param tensor_shape:
        :param num_queues: Number of memory regions. We need to switch between them because reading for inference &
          writing can happen in parallel.
        """

        # Initialize data strcutures that hold samples.
        self.sample_ids = []

        queue_shape = (capacity,) + tensor_shape
        self.x_queue = []
        self.x_att_queue = []
        for i in range(num_queues):
            self.x_queue.append(torch.empty(queue_shape, dtype=torch.int32, device='cuda'))
            self.x_att_queue.append(torch.empty(queue_shape, dtype=torch.int32, device='cuda'))

        # Queue state.
        self.capacity = capacity
        self.queue_len = 0
        self.cur_queue = 0
        self.num_queues = num_queues


    def add(self, sample_ids, samples):
        num_samples = len(sample_ids)
        idx = self.queue_len
        self.queue_len += num_samples
        assert self.queue_len < self.capacity

        self.sample_ids += sample_ids

        x, x_att = samples
        self.x_queue[self.cur_queue][idx:idx+num_samples].copy_(x, non_blocking=True)
        self.x_att_queue[self.cur_queue][idx:idx+num_samples].copy_(x_att, non_blocking=True)


    def get(self, num_samples=-1):
        """
        NOTE: For now this queue can only get all samples, it's not a full FIFO queue.
        This is problematic if the queue becomes so full that the resulting large batches
        leads to OOM. But if that happens, you're screwed anyways (huge latency), which is
        why we leave it like this for now.
        :param num_samples: Only there for compatible API.
        :return: (x tensor, x_att tensor), sample ids list
        """
        x = self.x_queue[self.cur_queue][:self.queue_len]
        x_att = self.x_att_queue[self.cur_queue][:self.queue_len]
        sample_ids = self.sample_ids
        self.sample_ids = []
        self.queue_len = 0
        self.cur_queue = (self.cur_queue + 1) % self.num_queues
        return (x, x_att), sample_ids


    def size(self):
        return self.queue_len


class LlamaQueue:
    """
    NOTE: For now, we use a different model queue for Llama since we just pass samples
    to ExLlama and don't move them to GPU ourselves. I'll change this in the future.
    """

    def __init__(self):
        self.samples = []
        self.sample_ids = []
        self.unissued_requests = 0 # Samples in the queue on which MultiGpuServer.infer() hasn't been called yet.
        self.counter = 0 # Count how often samples were added to use for triggering (instead of actual queue length)


    def add(self, sample_ids, samples):
        self.samples += samples
        self.sample_ids += sample_ids
        self.unissued_requests += len(sample_ids)
        self.counter += 1


    def get(self, num_samples):
        """
        if num_samples == -1: Get all.
        :param num_samples:
        :return: list of samples, list of sample ids
        """
        if num_samples == -1:
            q = self.samples
            q_ids = self.sample_ids
            self.samples = []
            self.sample_ids = []
            self.counter = 0
            return q, q_ids

        q = self.samples[:num_samples]
        q_ids = self.sample_ids[:num_samples]
        self.samples = self.samples[num_samples:]
        self.sample_ids = self.sample_ids[num_samples:]
        self.counter -= num_samples
        return q, q_ids


    def size(self):
        # return len(self.sample_ids)
        return self.counter


    def unissued_size(self):
        return self.unissued_requests



if __name__ == "__main__":
    q = BertQueue2(max_bs=4, num_queues=5)

    print("="*10)

    t = torch.ones((4,55))

    q.add(samples=(t,t), sample_ids=[1,2,3,4])

    print("="*10)

    q.add(samples=(t,t), sample_ids=[14,24,34,44])

    print("="*10)

    (t1, t2), s = q.get()

    print("="*10)

    print((t1, t2), s)
