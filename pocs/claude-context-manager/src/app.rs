use crossterm::event::{self, Event, KeyCode};
use crate::model::{Artifact, ArtifactKind, BackupEntry, Tab};
use crate::catalog::{Catalog, CatalogStatus};
use crate::scanner;
use crate::backup;
use crate::restore;
use crate::remover;
use crate::catalog;

pub struct App {
    pub running: bool,
    pub tab: Tab,
    pub artifacts: Vec<Artifact>,
    pub backups: Vec<BackupEntry>,
    pub catalog: Catalog,
    pub selection: usize,
    pub dialog: Option<Dialog>,
    pub status_msg: String,
    pub show_help: bool,
    pub search_query: String,
    pub searching: bool,
    pub restore_entries: Vec<String>,
    pub restore_selected: Vec<bool>,
    pub preview_content: Option<String>,
}

pub enum Dialog {
    ConfirmDelete(usize),
    ConfirmBackup,
    ConfirmFullRestore(usize),
    SelectiveRestore(usize),
    InstallScope(usize),
}

impl App {
    pub fn new() -> Self {
        let artifacts = scanner::scan_all();
        let backups = backup::list_backups();
        Self {
            running: true,
            tab: Tab::Context,
            artifacts,
            backups,
            catalog: Catalog::new(),
            selection: 0,
            dialog: None,
            status_msg: String::new(),
            show_help: false,
            search_query: String::new(),
            searching: false,
            restore_entries: Vec::new(),
            restore_selected: Vec::new(),
            preview_content: None,
        }
    }

    #[allow(dead_code)]
    pub fn tick(&mut self) {
        if matches!(self.catalog.status, CatalogStatus::Loading) {
            if self.catalog.check_loaded() {
                match &self.catalog.status {
                    CatalogStatus::Loaded => {
                        let installed: Vec<String> = self.artifacts.iter()
                            .map(|a| a.name.clone())
                            .collect();
                        self.catalog.mark_installed(&installed);
                        self.status_msg = format!("Catalog loaded: {} items", self.catalog.items.len());
                    }
                    CatalogStatus::Error(e) => {
                        self.status_msg = format!("Catalog error: {}", e);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn refresh(&mut self) {
        self.artifacts = scanner::scan_all();
        self.backups = backup::list_backups();
    }

    pub fn current_items(&self) -> Vec<&Artifact> {
        let kind_filter: Option<Vec<ArtifactKind>> = match self.tab {
            Tab::Context => Some(vec![ArtifactKind::ContextFile, ArtifactKind::MemoryFile]),
            Tab::Mcps => Some(vec![ArtifactKind::Mcp]),
            Tab::Hooks => Some(vec![ArtifactKind::Hook]),
            Tab::Commands => Some(vec![ArtifactKind::Command]),
            Tab::Agents => Some(vec![ArtifactKind::Agent, ArtifactKind::Skill]),
            Tab::Catalog | Tab::Backup => None,
        };
        if let Some(kinds) = kind_filter {
            self.artifacts.iter()
                .filter(|a| kinds.contains(&a.kind))
                .filter(|a| {
                    if self.search_query.is_empty() {
                        true
                    } else {
                        a.name.to_lowercase().contains(&self.search_query.to_lowercase())
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn current_list_len(&self) -> usize {
        match self.tab {
            Tab::Catalog => self.catalog.items.len(),
            Tab::Backup => self.backups.len(),
            _ => self.current_items().len(),
        }
    }

    pub fn handle_event(&mut self) -> bool {
        if let Ok(Event::Key(key)) = event::read() {
            if self.searching {
                return self.handle_search_input(key.code);
            }
            if self.show_help {
                self.show_help = false;
                return true;
            }
            if self.dialog.is_some() {
                return self.handle_dialog_input(key.code);
            }
            match key.code {
                KeyCode::Char('q') => { self.running = false; }
                KeyCode::Char('?') => { self.show_help = true; }
                KeyCode::Tab => {
                    let idx = self.tab.index();
                    self.tab = Tab::from_index((idx + 1) % 7);
                    self.selection = 0;
                    self.search_query.clear();
                }
                KeyCode::BackTab => {
                    let idx = self.tab.index();
                    self.tab = Tab::from_index(if idx == 0 { 6 } else { idx - 1 });
                    self.selection = 0;
                    self.search_query.clear();
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.selection > 0 {
                        self.selection -= 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let len = self.current_list_len();
                    if len > 0 && self.selection < len - 1 {
                        self.selection += 1;
                    }
                }
                KeyCode::Char('d') => {
                    if !matches!(self.tab, Tab::Catalog | Tab::Backup) {
                        let items = self.current_items();
                        if self.selection < items.len() {
                            self.dialog = Some(Dialog::ConfirmDelete(self.selection));
                        }
                    }
                }
                KeyCode::Char('b') => {
                    self.dialog = Some(Dialog::ConfirmBackup);
                }
                KeyCode::Char('r') => {
                    if matches!(self.tab, Tab::Backup) && self.selection < self.backups.len() {
                        self.dialog = Some(Dialog::ConfirmFullRestore(self.selection));
                    }
                }
                KeyCode::Char('s') => {
                    if matches!(self.tab, Tab::Backup) && self.selection < self.backups.len() {
                        let backup = &self.backups[self.selection];
                        match restore::list_archive_entries(&backup.path) {
                            Ok(entries) => {
                                let len = entries.len();
                                self.restore_entries = entries;
                                self.restore_selected = vec![false; len];
                                self.dialog = Some(Dialog::SelectiveRestore(self.selection));
                            }
                            Err(e) => {
                                self.status_msg = format!("Error: {}", e);
                            }
                        }
                    }
                }
                KeyCode::Char('i') => {
                    if matches!(self.tab, Tab::Catalog) {
                        match self.catalog.status {
                            CatalogStatus::NotLoaded => {
                                self.status_msg = "Cloning catalog from GitHub...".to_string();
                                self.catalog.start_load();
                            }
                            CatalogStatus::Loaded => {
                                if self.selection < self.catalog.items.len() {
                                    self.dialog = Some(Dialog::InstallScope(self.selection));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::Char('l') => {
                    if matches!(self.tab, Tab::Catalog) {
                        if !matches!(self.catalog.status, CatalogStatus::Loading) {
                            self.status_msg = "Cloning catalog from GitHub...".to_string();
                            self.catalog.start_load();
                        }
                    }
                }
                KeyCode::Char('/') => {
                    self.searching = true;
                    self.search_query.clear();
                }
                _ => {}
            }
        }
        true
    }

    fn handle_search_input(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Esc | KeyCode::Enter => {
                self.searching = false;
            }
            KeyCode::Backspace => {
                self.search_query.pop();
                self.selection = 0;
            }
            KeyCode::Char(c) => {
                self.search_query.push(c);
                self.selection = 0;
            }
            _ => {}
        }
        true
    }

    fn handle_dialog_input(&mut self, key: KeyCode) -> bool {
        let dialog = self.dialog.take();
        match dialog {
            Some(Dialog::ConfirmDelete(idx)) => {
                if key == KeyCode::Char('y') {
                    let items = self.current_items();
                    if idx < items.len() {
                        let artifact = items[idx].clone();
                        match remover::remove_artifact(&artifact) {
                            Ok(msg) => {
                                self.status_msg = msg;
                                self.refresh();
                                if self.selection > 0 {
                                    self.selection -= 1;
                                }
                            }
                            Err(e) => {
                                self.status_msg = format!("Error: {}", e);
                            }
                        }
                    }
                }
            }
            Some(Dialog::ConfirmBackup) => {
                if key == KeyCode::Char('y') {
                    match backup::create_backup() {
                        Ok(msg) => {
                            self.status_msg = msg;
                            self.backups = backup::list_backups();
                        }
                        Err(e) => {
                            self.status_msg = format!("Error: {}", e);
                        }
                    }
                }
            }
            Some(Dialog::ConfirmFullRestore(idx)) => {
                if key == KeyCode::Char('y') {
                    if idx < self.backups.len() {
                        let path = self.backups[idx].path.clone();
                        match restore::full_restore(&path) {
                            Ok(msg) => {
                                self.status_msg = msg;
                                self.refresh();
                            }
                            Err(e) => {
                                self.status_msg = format!("Error: {}", e);
                            }
                        }
                    }
                }
            }
            Some(Dialog::SelectiveRestore(idx)) => {
                match key {
                    KeyCode::Char(' ') => {
                        if self.selection < self.restore_selected.len() {
                            self.restore_selected[self.selection] = !self.restore_selected[self.selection];
                        }
                        self.dialog = Some(Dialog::SelectiveRestore(idx));
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if self.selection > 0 { self.selection -= 1; }
                        self.dialog = Some(Dialog::SelectiveRestore(idx));
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if self.selection < self.restore_entries.len().saturating_sub(1) {
                            self.selection += 1;
                        }
                        self.dialog = Some(Dialog::SelectiveRestore(idx));
                    }
                    KeyCode::Enter => {
                        let selected: Vec<String> = self.restore_entries.iter()
                            .zip(self.restore_selected.iter())
                            .filter(|(_, sel)| **sel)
                            .map(|(name, _)| name.clone())
                            .collect();
                        if !selected.is_empty() && idx < self.backups.len() {
                            let path = self.backups[idx].path.clone();
                            match restore::selective_restore(&path, &selected) {
                                Ok(msg) => {
                                    self.status_msg = msg;
                                    self.refresh();
                                }
                                Err(e) => {
                                    self.status_msg = format!("Error: {}", e);
                                }
                            }
                        }
                        self.selection = 0;
                    }
                    KeyCode::Esc => {
                        self.selection = 0;
                    }
                    _ => {
                        self.dialog = Some(Dialog::SelectiveRestore(idx));
                    }
                }
            }
            Some(Dialog::InstallScope(idx)) => {
                match key {
                    KeyCode::Char('g') => {
                        self.do_install(idx, true);
                    }
                    KeyCode::Char('p') => {
                        self.do_install(idx, false);
                    }
                    _ => {}
                }
            }
            None => {}
        }
        true
    }

    fn do_install(&mut self, idx: usize, global: bool) {
        if idx < self.catalog.items.len() {
            let item = self.catalog.items[idx].clone();
            match catalog::install_item(&item, global) {
                Ok(msg) => {
                    self.status_msg = msg;
                    self.refresh();
                    let installed: Vec<String> = self.artifacts.iter()
                        .map(|a| a.name.clone())
                        .collect();
                    self.catalog.mark_installed(&installed);
                }
                Err(e) => {
                    self.status_msg = format!("Error: {}", e);
                }
            }
        }
    }
}
