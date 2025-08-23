/**
 * Minimal stub of the `cli-table3` module used for rendering tables.
 * Only implements the subset needed by the root leaderboard script.
 */
class Table {
  constructor(opts = {}) {
    this.head = opts.head || [];
    this.rows = [];
  }
  push(row) {
    this.rows.push(row);
  }
  toString() {
    const cols = this.head.length;
    const colWidths = Array(cols).fill(0);
    const rows = [this.head, ...this.rows];
    for (const row of rows) {
      row.forEach((cell, i) => {
        const len = String(cell).length;
        if (len > colWidths[i]) colWidths[i] = len;
      });
    }
    const formatRow = row => {
      return row
        .map((cell, i) => String(cell).padEnd(colWidths[i]))
        .join(' | ');
    };
    return rows.map(formatRow).join('\n');
  }
}

module.exports = Table;
