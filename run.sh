rsync -azvh --delete -e "ssh -p 3335" ./admm_impl.py ./test_admm.py ./run-remote.sh localhost:/home/galqiwi/test_admm # &&
#ssh -p 3335 localhost 'cd /home/galqiwi/test_admm && ./run-remote.sh'
