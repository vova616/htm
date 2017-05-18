pub struct Column {
    pub index: usize,
}

pub struct ColumnHandle(usize);

impl Column {
    pub fn new(cells: usize, index: usize) -> Column {
        Column { index: index }
    }
}
