import threading
import time
from update_worker import update_worker
from server import start_uvicorn

if __name__ == "__main__":
    # Create two daemon threads
    api_thread = threading.Thread(
        target=start_uvicorn,
        daemon=True,
    )
    update_worker_thread = threading.Thread(
        target=update_worker,
        daemon=True,
    )

    # Start the threads
    api_thread.start()
    update_worker_thread.start()

    try:
        while True:
            # Check if both threads are alive
            if not api_thread.is_alive() or not update_worker_thread.is_alive():
                print("One of the threads has stopped. Exiting the main loop.")
                break
            time.sleep(0.5)  # Check every half second
    except KeyboardInterrupt:
        print("Main thread interrupted.")

    print("Main thread exiting.")
