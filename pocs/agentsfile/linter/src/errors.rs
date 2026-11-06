use agentsfile::{Issue, IssueKind};

use crate::diagnostic::Diagnostic;

pub fn from_issue(issue: &Issue) -> Diagnostic {
    let code = match issue.kind {
        IssueKind::UnknownAttribute => "E002",
        IssueKind::UnknownBlock => "E003",
        IssueKind::DuplicateAttribute => "E004",
        IssueKind::WrongKind => "E005",
        IssueKind::InvalidValue => "E006",
        IssueKind::DuplicateName => "E007",
        IssueKind::MissingRequired => "E008",
        IssueKind::TransportRequirement => "E009",
        IssueKind::Conflict => "E010",
    };
    Diagnostic::error(code, issue.error.line, issue.error.col, issue.error.message.clone())
}

pub fn from_syntax(error: &agentsfile::Error) -> Diagnostic {
    Diagnostic::error("E001", error.line, error.col, error.message.clone())
}
