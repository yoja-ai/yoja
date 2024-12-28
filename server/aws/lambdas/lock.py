import os
import fcntl
import time

def lock_user_local(index_dir):
    lockfile_path = os.path.join(index_dir, 'lockfile')

    # Create the lockfile if it does not exist
    if not os.path.exists(lockfile_path):
        with open(lockfile_path, 'w') as lockfile:
            lockfile.write('')  # Create an empty file

    # Open the lockfile
    lockfile = open(lockfile_path, 'r')

    try:
        # Acquire an exclusive lock on the lockfile
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Lock acquired on {lockfile_path}")
        return True
    except IOError as e:
        print(f"Failed to acquire lock: {e}")
        return False

def unlock_user_local(index_dir):
    lockfile_path = os.path.join(index_dir, 'lockfile')
    lockfile = open(lockfile_path, 'r')
    # Release the lock and close the file
    fcntl.flock(lockfile, fcntl.LOCK_UN)
    lockfile.close()
    print(f"Lock released on {lockfile_path}")

