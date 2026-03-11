SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS trajectory_logs;
DROP TABLE IF EXISTS memory_logs;
DROP TABLE IF EXISTS retrieval_logs;
DROP TABLE IF EXISTS task_runs;

SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE IF NOT EXISTS task_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(64) NOT NULL COMMENT '实验ID，用于实验隔离',
    task_id VARCHAR(64) NOT NULL COMMENT '任务ID，如 task_001',
    task_order INT NOT NULL COMMENT '任务顺序，用于历史过滤',
    query_text TEXT NOT NULL COMMENT '任务输入问题',
    task_start_time DATETIME NOT NULL COMMENT '任务开始时间，用于 created_at 时序约束',
    final_answer LONGTEXT DEFAULT NULL COMMENT '最终答案',
    success_flag TINYINT DEFAULT 0 COMMENT '任务是否成功完成',
    reward_score FLOAT DEFAULT NULL COMMENT '奖励分数（RL阶段预留）',
    token_cost INT DEFAULT NULL COMMENT 'token消耗（可选）',
    latency_ms INT DEFAULT NULL COMMENT '耗时毫秒',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '日志创建时间',

    INDEX idx_task_runs_experiment_id (experiment_id),
    INDEX idx_task_runs_task_id (task_id),
    INDEX idx_task_runs_task_order (task_order),
    INDEX idx_task_runs_experiment_task_order (experiment_id, task_order),
    INDEX idx_task_runs_task_start_time (task_start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务运行主表';


CREATE TABLE IF NOT EXISTS retrieval_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    task_run_id BIGINT NOT NULL COMMENT '关联 task_runs.id',
    query_text TEXT NOT NULL COMMENT '检索查询',
    doc_id VARCHAR(128) NOT NULL COMMENT '文档ID',
    chunk_id VARCHAR(128) DEFAULT NULL COMMENT '分块ID',
    rank_pos INT NOT NULL COMMENT '排序位置',
    score FLOAT DEFAULT NULL COMMENT '检索分数',
    used_flag TINYINT DEFAULT 0 COMMENT '是否被最终使用',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '日志创建时间',

    INDEX idx_retrieval_logs_task_run_id (task_run_id),
    INDEX idx_retrieval_logs_doc_id (doc_id),
    INDEX idx_retrieval_logs_chunk_id (chunk_id),

    CONSTRAINT fk_retrieval_logs_task_run
        FOREIGN KEY (task_run_id) REFERENCES task_runs(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识检索日志表';


CREATE TABLE IF NOT EXISTS memory_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    task_run_id BIGINT NOT NULL COMMENT '关联 task_runs.id',
    memory_key VARCHAR(128) NOT NULL COMMENT 'memory日志键，如 task:task_001:retrieve',
    operation_type VARCHAR(32) NOT NULL COMMENT 'retrieve/write',
    memory_content LONGTEXT DEFAULT NULL COMMENT 'memory内容',
    relevance_score FLOAT DEFAULT NULL COMMENT '相关性分数',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '日志创建时间',

    INDEX idx_memory_logs_task_run_id (task_run_id),
    INDEX idx_memory_logs_memory_key (memory_key),
    INDEX idx_memory_logs_operation_type (operation_type),

    CONSTRAINT fk_memory_logs_task_run
        FOREIGN KEY (task_run_id) REFERENCES task_runs(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='memory读写日志表';


CREATE TABLE IF NOT EXISTS trajectory_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    task_run_id BIGINT NOT NULL COMMENT '关联 task_runs.id',
    step_no INT NOT NULL COMMENT '步骤编号',
    agent_name VARCHAR(64) NOT NULL COMMENT 'Agent名称',
    action_type VARCHAR(64) NOT NULL COMMENT '动作类型',
    action_input LONGTEXT DEFAULT NULL COMMENT '动作输入',
    action_output LONGTEXT DEFAULT NULL COMMENT '动作输出',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '日志创建时间',

    INDEX idx_trajectory_logs_task_run_id (task_run_id),
    INDEX idx_trajectory_logs_step_no (step_no),
    INDEX idx_trajectory_logs_agent_name (agent_name),
    INDEX idx_trajectory_logs_action_type (action_type),

    CONSTRAINT fk_trajectory_logs_task_run
        FOREIGN KEY (task_run_id) REFERENCES task_runs(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Agent轨迹日志表';