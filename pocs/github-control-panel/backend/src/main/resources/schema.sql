CREATE TABLE IF NOT EXISTS repository (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    owner VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    full_name VARCHAR(512) NOT NULL UNIQUE,
    added_at DATETIME NOT NULL,
    last_synced_at DATETIME NULL
);

CREATE TABLE IF NOT EXISTS pull_request (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    repository_id BIGINT NOT NULL,
    number INT NOT NULL,
    title TEXT NOT NULL,
    author VARCHAR(255) NULL,
    state VARCHAR(32) NOT NULL,
    draft BOOLEAN NOT NULL,
    ci_status VARCHAR(32) NULL,
    mergeable VARCHAR(32) NULL,
    review_decision VARCHAR(32) NULL,
    review_requests_count INT NOT NULL DEFAULT 0,
    labels TEXT NULL,
    reviewers TEXT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    closed_at DATETIME NULL,
    merged_at DATETIME NULL,
    url TEXT NOT NULL,
    INDEX idx_pr_repo (repository_id),
    INDEX idx_pr_state (state)
);

CREATE TABLE IF NOT EXISTS issue (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    repository_id BIGINT NOT NULL,
    number INT NOT NULL,
    title TEXT NOT NULL,
    author VARCHAR(255) NULL,
    state VARCHAR(32) NOT NULL,
    body MEDIUMTEXT NULL,
    comments_count INT NOT NULL DEFAULT 0,
    assignees_count INT NOT NULL DEFAULT 0,
    assignees TEXT NULL,
    labels TEXT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    closed_at DATETIME NULL,
    url TEXT NOT NULL,
    INDEX idx_issue_repo (repository_id),
    INDEX idx_issue_state (state)
);

CREATE TABLE IF NOT EXISTS contribution (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    repository_id BIGINT NOT NULL,
    login VARCHAR(255) NOT NULL,
    commits INT NOT NULL DEFAULT 0,
    prs_opened INT NOT NULL DEFAULT 0,
    issues_opened INT NOT NULL DEFAULT 0,
    reviews INT NOT NULL DEFAULT 0,
    INDEX idx_contrib_repo (repository_id),
    INDEX idx_contrib_login (login)
);
