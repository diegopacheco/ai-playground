use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use crate::ui::{DialogState, NewSessionDialog};

pub enum InputResult {
    None,
    Quit,
    NewSession,
    FocusTerminal,
    ToggleFocus,
    SwitchSession(usize),
    CreateSession,
    KillSession,
    NavigateUp,
    NavigateDown,
    Select,
    Cancel,
    Back,
    TerminalInput(Vec<u8>),
}

pub fn handle_input(
    key: KeyEvent,
    dialog: &mut NewSessionDialog,
    terminal_focused: bool,
) -> InputResult {
    let ctrl_only = key.modifiers == KeyModifiers::CONTROL;
    let cmd_only = key.modifiers == KeyModifiers::SUPER;

    if cmd_only {
        match key.code {
            KeyCode::Char('t') => return InputResult::NewSession,
            KeyCode::Char('e') => return InputResult::FocusTerminal,
            _ => {}
        }
    }

    if ctrl_only {
        match key.code {
            KeyCode::Char('w') => return InputResult::Quit,
            KeyCode::Char('1') => return InputResult::SwitchSession(0),
            KeyCode::Char('2') => return InputResult::SwitchSession(1),
            KeyCode::Char('3') => return InputResult::SwitchSession(2),
            KeyCode::Char('4') => return InputResult::SwitchSession(3),
            KeyCode::Char('5') => return InputResult::SwitchSession(4),
            KeyCode::Char('6') => return InputResult::SwitchSession(5),
            KeyCode::Char('7') => return InputResult::SwitchSession(6),
            KeyCode::Char('8') => return InputResult::SwitchSession(7),
            KeyCode::Char('9') => return InputResult::SwitchSession(8),
            _ => {}
        }
    }

    if dialog.is_open() {
        return handle_dialog_input(key, dialog);
    }

    if terminal_focused {
        return handle_terminal_input(key);
    }

    handle_list_input(key)
}

fn handle_dialog_input(key: KeyEvent, dialog: &mut NewSessionDialog) -> InputResult {
    match key.code {
        KeyCode::Esc => {
            if dialog.state == DialogState::SelectingDirectory {
                dialog.back_to_agent();
            } else {
                dialog.close();
            }
            InputResult::Cancel
        }
        KeyCode::Up => {
            match dialog.state {
                DialogState::SelectingAgent => dialog.prev_agent(),
                DialogState::SelectingDirectory => dialog.file_browser.up(),
                _ => {}
            }
            InputResult::NavigateUp
        }
        KeyCode::Down => {
            match dialog.state {
                DialogState::SelectingAgent => dialog.next_agent(),
                DialogState::SelectingDirectory => dialog.file_browser.down(),
                _ => {}
            }
            InputResult::NavigateDown
        }
        KeyCode::Enter => {
            match dialog.state {
                DialogState::SelectingAgent => {
                    dialog.confirm_agent();
                    InputResult::Select
                }
                DialogState::SelectingDirectory => {
                    dialog.file_browser.enter();
                    InputResult::Select
                }
                _ => InputResult::None
            }
        }
        KeyCode::Backspace => {
            if dialog.state == DialogState::SelectingDirectory {
                dialog.file_browser.go_up();
            }
            InputResult::Back
        }
        KeyCode::Tab => {
            if dialog.state == DialogState::SelectingDirectory {
                return InputResult::CreateSession;
            }
            InputResult::None
        }
        KeyCode::Char(' ') => {
            if dialog.state == DialogState::SelectingDirectory {
                return InputResult::CreateSession;
            }
            InputResult::None
        }
        _ => InputResult::None
    }
}

fn handle_terminal_input(key: KeyEvent) -> InputResult {
    match key.code {
        KeyCode::Esc => InputResult::Cancel,
        KeyCode::Tab => InputResult::ToggleFocus,
        KeyCode::Char(c) => InputResult::TerminalInput(c.to_string().into_bytes()),
        KeyCode::Enter => InputResult::TerminalInput(vec![b'\r']),
        KeyCode::Backspace => InputResult::TerminalInput(vec![127]),
        KeyCode::Up => InputResult::TerminalInput(vec![27, 91, 65]),
        KeyCode::Down => InputResult::TerminalInput(vec![27, 91, 66]),
        KeyCode::Right => InputResult::TerminalInput(vec![27, 91, 67]),
        KeyCode::Left => InputResult::TerminalInput(vec![27, 91, 68]),
        _ => InputResult::None,
    }
}

fn handle_list_input(key: KeyEvent) -> InputResult {
    match key.code {
        KeyCode::Up => InputResult::NavigateUp,
        KeyCode::Down => InputResult::NavigateDown,
        KeyCode::Enter => InputResult::Select,
        KeyCode::Tab => InputResult::ToggleFocus,
        KeyCode::Char('q') => InputResult::KillSession,
        KeyCode::Char('c') => InputResult::NewSession,
        KeyCode::Esc => InputResult::Cancel,
        _ => InputResult::None,
    }
}
