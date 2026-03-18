class NullLogger:
    def close(self):
        pass

    def log_task_run(self, *args, **kwargs):
        return 0

    def update_task_result(self, *args, **kwargs):
        pass

    def log_retrieval(self, *args, **kwargs):
        return 0

    def log_memory(self, *args, **kwargs):
        return 0

    def log_trajectory(self, *args, **kwargs):
        return 0
