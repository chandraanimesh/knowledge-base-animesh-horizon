import React, { useState } from 'react';
import { TableVisualization } from './types';

interface DataTableProps {
  spec: TableVisualization;
}

export const DataTable: React.FC<DataTableProps> = ({ spec }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  const columns = spec.columns || [];
  const rawRows = spec.rows || [];

  if (columns.length === 0 || rawRows.length === 0) return null;

  // Filter rows
  const filteredRows = rawRows.filter((r) => {
    if (!searchTerm.trim()) return true;
    const term = searchTerm.toLowerCase();
    return columns.some((c) => {
      const val = r[c.key];
      return val !== undefined && val !== null && String(val).toLowerCase().includes(term);
    });
  });

  // Sort rows
  const sortedRows = [...filteredRows].sort((a, b) => {
    if (!sortKey) return 0;
    const valA = a[sortKey];
    const valB = b[sortKey];
    if (valA === valB) return 0;
    if (valA === undefined || valA === null) return 1;
    if (valB === undefined || valB === null) return -1;
    if (typeof valA === 'number' && typeof valB === 'number') {
      return sortAsc ? valA - valB : valB - valA;
    }
    return sortAsc
      ? String(valA).localeCompare(String(valB))
      : String(valB).localeCompare(String(valA));
  });

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const formatCellValue = (val: any, col: (typeof columns)[0]) => {
    if (val === undefined || val === null) return '—';
    if (col.type === 'percent') {
      return `${val}%`;
    }
    if (col.type === 'number' || col.type === 'integer') {
      const formatted = typeof val === 'number' ? val.toLocaleString() : val;
      return col.unit ? `${formatted} ${col.unit}` : formatted;
    }
    return String(val);
  };

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {/* Search Header if > 4 rows */}
      {rawRows.length > 4 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '4px' }}>
          <input
            type="text"
            placeholder="Search table rows…"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              padding: '4px 10px',
              fontSize: '12px',
              borderRadius: '6px',
              border: '1px solid #CBD5E1',
              background: '#FFFFFF',
              color: '#334155',
              outline: 'none',
              width: '180px',
            }}
          />
        </div>
      )}

      {/* Table Container */}
      <div
        style={{
          width: '100%',
          overflowX: 'auto',
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          background: '#FFFFFF',
        }}
      >
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12.5px', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: '#F8FAFC', borderBottom: '1px solid #E2E8F0' }}>
              {columns.map((c) => (
                <th
                  key={c.key}
                  onClick={() => handleSort(c.key)}
                  style={{
                    padding: '10px 14px',
                    fontWeight: 600,
                    color: '#475569',
                    cursor: 'pointer',
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span>{c.label}</span>
                    {sortKey === c.key ? (
                      <span style={{ fontSize: '10px', color: '#2563EB' }}>{sortAsc ? '▲' : '▼'}</span>
                    ) : (
                      <span style={{ fontSize: '9px', color: '#CBD5E1' }}>⇅</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} style={{ padding: '24px', textAlign: 'center', color: '#94A3B8' }}>
                  No matching data
                </td>
              </tr>
            ) : (
              sortedRows.map((row, rIdx) => (
                <tr
                  key={rIdx}
                  style={{
                    borderBottom: rIdx === sortedRows.length - 1 ? 'none' : '1px solid #F1F5F9',
                    background: rIdx % 2 === 0 ? '#FFFFFF' : '#FAFAFA',
                    transition: 'background 0.1s',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = '#F1F5F9')}
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background = rIdx % 2 === 0 ? '#FFFFFF' : '#FAFAFA')
                  }
                >
                  {columns.map((c) => {
                    const isNumeric = c.type === 'number' || c.type === 'integer' || c.type === 'percent';
                    return (
                      <td
                        key={c.key}
                        style={{
                          padding: '9px 14px',
                          color: isNumeric ? '#1E293B' : '#334155',
                          fontWeight: isNumeric ? 500 : 400,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {formatCellValue(row[c.key], c)}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Row count summary */}
      <div style={{ fontSize: '11px', color: '#94A3B8', textAlign: 'right' }}>
        Showing {sortedRows.length} of {rawRows.length} record{rawRows.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
};
