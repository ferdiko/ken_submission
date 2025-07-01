import asyncio
import time

# Example async function
async def async_function(name, delay):
    print(f"{name}: Preparing to contact another process.")
    await asyncio.sleep(delay)  # Simulating waiting for a response
    print(f"{name}: Received response from another process.")
    return f"{name}: Done"

# Helper to step the event loop until all tasks reach their first await
async def run_step_until_await(tasks):
    await asyncio.sleep(0)  # Let the event loop step once
    await asyncio.gather(*tasks)  # Gather ensures they are at least started

# Main program
if __name__ == "__main__":
    # Create an event loop
    loop = asyncio.get_event_loop()

    # Schedule tasks (but don't block yet)
    task1 = loop.create_task(async_function("Task1", 3))
    task2 = loop.create_task(async_function("Task2", 2))

    # Ensure tasks execute only up to their first `await`
    loop.run_until_complete(run_step_until_await([task1, task2]))

    print("Back in main program, async tasks are waiting for responses...")

    # Do other work in the main program
    print(f"Main program doing other things...")
    time.sleep(1)
    print(f"Main loop done...")

    # Now wait for the tasks to complete and get results
    results = loop.run_until_complete(asyncio.gather(task1, task2))

    print("Results:")
    for result in results:
        print(result)

    # Close the loop
    loop.close()
