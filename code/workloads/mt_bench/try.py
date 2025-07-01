import asyncio
import threading
import time

# Example async function that interacts with another process
async def async_function(name, delay):
    print(f"{name}: Preparing to contact another process.")
    await asyncio.sleep(delay)  # Simulating waiting for a response
    print(f"{name}: Received response from another process.")
    return f"{name}: Done"

# Function to run the event loop in a thread
def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Function to start the coroutine until the first await
def start_coroutine_until_await(coro, loop):
    # Schedule the coroutine on the event loop and return the Future
    return asyncio.run_coroutine_threadsafe(coro, loop)

if __name__ == "__main__":
    start = time.time()


    # Create a new event loop
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=start_loop, args=(loop,), daemon=True)
    loop_thread.start()

    # Schedule the async function
    future1 = start_coroutine_until_await(async_function("Task1", 4), loop)
    future2 = start_coroutine_until_await(async_function("Task2", 2), loop)

    print("Back in main program, async tasks are waiting for responses...")

    # Do other work in the main program while tasks are awaiting
    time.sleep(3)
    print("Main program doing other things...")

    # Wait for tasks to complete and get results
    result1 = future1.result()  # Blocks until Task1 is complete
    result2 = future2.result()  # Blocks until Task2 is complete

    print("Results:")
    print(result1)
    print(result2)

    # Clean up
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join()

    print("TIME:", time.time() - start)
