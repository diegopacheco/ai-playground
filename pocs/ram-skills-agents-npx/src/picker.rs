pub struct Entry {
    pub name: String,
    pub haystack: String,
}

pub struct Picker {
    entries: Vec<Entry>,
    selected: Vec<bool>,
    query: String,
    visible: Vec<usize>,
    cursor: usize,
}

impl Picker {
    pub fn new(entries: Vec<Entry>) -> Self {
        let selected = vec![false; entries.len()];
        let mut picker = Picker { entries, selected, query: String::new(), visible: Vec::new(), cursor: 0 };
        picker.refilter();
        picker
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn visible(&self) -> &[usize] {
        &self.visible
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_selected(&self, index: usize) -> bool {
        self.selected[index]
    }

    pub fn chosen(&self) -> Vec<usize> {
        (0..self.selected.len()).filter(|i| self.selected[*i]).collect()
    }

    pub fn type_char(&mut self, c: char) {
        self.query.push(c);
        self.refilter();
    }

    pub fn backspace(&mut self) {
        self.query.pop();
        self.refilter();
    }

    pub fn clear_query(&mut self) {
        self.query.clear();
        self.refilter();
    }

    pub fn up(&mut self) {
        if !self.visible.is_empty() {
            self.cursor = (self.cursor + self.visible.len() - 1) % self.visible.len();
        }
    }

    pub fn down(&mut self) {
        if !self.visible.is_empty() {
            self.cursor = (self.cursor + 1) % self.visible.len();
        }
    }

    pub fn set_cursor(&mut self, row: usize) {
        if row < self.visible.len() {
            self.cursor = row;
        }
    }

    pub fn toggle(&mut self) {
        if let Some(&index) = self.visible.get(self.cursor) {
            self.selected[index] = !self.selected[index];
        }
    }

    pub fn toggle_all_visible(&mut self) {
        let all = self.visible.iter().all(|&i| self.selected[i]);
        for &index in &self.visible {
            self.selected[index] = !all;
        }
    }

    fn refilter(&mut self) {
        let lowered = self.query.to_lowercase();
        let terms: Vec<&str> = lowered.split_whitespace().collect();
        let mut visible: Vec<usize> = (0..self.entries.len())
            .filter(|&i| terms.iter().all(|term| self.entries[i].haystack.contains(term)))
            .collect();
        visible.sort_by_key(|&i| self.rank(i, &terms));
        self.visible = visible;
        self.cursor = 0;
    }

    fn rank(&self, index: usize, terms: &[&str]) -> u8 {
        let name = self.entries[index].name.to_lowercase();
        match terms.first() {
            Some(term) if name.starts_with(term) => 0,
            Some(term) if name.contains(term) => 1,
            Some(_) => 2,
            None => 0,
        }
    }
}
