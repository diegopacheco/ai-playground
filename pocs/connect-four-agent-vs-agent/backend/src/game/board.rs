use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Cell {
    Empty,
    X,
    O,
}

impl Cell {
    pub fn to_char(&self) -> char {
        match self {
            Cell::Empty => '.',
            Cell::X => 'X',
            Cell::O => 'O',
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Board {
    cells: [[Cell; 7]; 6],
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: [[Cell::Empty; 7]; 6],
        }
    }

    pub fn drop_piece(&mut self, column: u8, player: Cell) -> Result<u8, String> {
        if column > 6 {
            return Err("Invalid column".to_string());
        }
        for row in 0..6 {
            if self.cells[row][column as usize] == Cell::Empty {
                self.cells[row][column as usize] = player;
                return Ok(row as u8);
            }
        }
        Err("Column is full".to_string())
    }

    pub fn is_column_valid(&self, column: u8) -> bool {
        if column > 6 {
            return false;
        }
        self.cells[5][column as usize] == Cell::Empty
    }

    pub fn check_winner(&self) -> Option<Cell> {
        for row in 0..6 {
            for col in 0..4 {
                let cell = self.cells[row][col];
                if cell != Cell::Empty
                    && cell == self.cells[row][col + 1]
                    && cell == self.cells[row][col + 2]
                    && cell == self.cells[row][col + 3]
                {
                    return Some(cell);
                }
            }
        }
        for row in 0..3 {
            for col in 0..7 {
                let cell = self.cells[row][col];
                if cell != Cell::Empty
                    && cell == self.cells[row + 1][col]
                    && cell == self.cells[row + 2][col]
                    && cell == self.cells[row + 3][col]
                {
                    return Some(cell);
                }
            }
        }
        for row in 0..3 {
            for col in 0..4 {
                let cell = self.cells[row][col];
                if cell != Cell::Empty
                    && cell == self.cells[row + 1][col + 1]
                    && cell == self.cells[row + 2][col + 2]
                    && cell == self.cells[row + 3][col + 3]
                {
                    return Some(cell);
                }
            }
        }
        for row in 3..6 {
            for col in 0..4 {
                let cell = self.cells[row][col];
                if cell != Cell::Empty
                    && cell == self.cells[row - 1][col + 1]
                    && cell == self.cells[row - 2][col + 2]
                    && cell == self.cells[row - 3][col + 3]
                {
                    return Some(cell);
                }
            }
        }
        None
    }

    pub fn is_full(&self) -> bool {
        for col in 0..7 {
            if self.cells[5][col] == Cell::Empty {
                return false;
            }
        }
        true
    }

    pub fn to_text(&self) -> String {
        let mut result = String::new();
        for row in (0..6).rev() {
            for col in 0..7 {
                result.push(self.cells[row][col].to_char());
                if col < 6 {
                    result.push(' ');
                }
            }
            result.push('\n');
        }
        result
    }

    pub fn to_array(&self) -> [[char; 7]; 6] {
        let mut arr = [['.'; 7]; 6];
        for row in 0..6 {
            for col in 0..7 {
                arr[5 - row][col] = self.cells[row][col].to_char();
            }
        }
        arr
    }

    pub fn get_cells(&self) -> &[[Cell; 7]; 6] {
        &self.cells
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}
