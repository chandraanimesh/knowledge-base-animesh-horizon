import React, { useState } from 'react';
import { PieVisualization } from './types';

interface PieChartProps {
  spec: PieVisualization;
}

const PALETTE = [
  '#2563EB', // Blue
  '#10B981', // Emerald
  '#F59E0B', // Amber
  '#8B5CF6', // Purple
  '#EC4899', // Pink
  '#06B6D4', // Cyan
  '#F97316', // Orange
  '#6366F1', // Indigo
];

export const PieChart: React.FC<PieChartProps> = ({ spec }) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const data = spec.data || [];
  if (data.length < 2) return null;

  const total = data.reduce((sum, d) => sum + (typeof d.value === 'number' ? Math.max(0, d.value) : 0), 0);
  if (total <= 0) return null;

  const unit = spec.unit ? ` ${spec.unit}` : '';

  // SVG Donut parameters
  const size = 260;
  const center = size / 2;
  const radius = 100;
  const innerRadius = 55;

  let currentAngle = -Math.PI / 2;

  const slices = data.map((d, idx) => {
    const val = Math.max(0, d.value);
    const sliceAngle = (val / total) * 2 * Math.PI;
    const startAngle = currentAngle;
    const endAngle = currentAngle + sliceAngle;
    currentAngle = endAngle;

    const isHovered = hoveredIndex === idx;
    const r = isHovered ? radius + 5 : radius;
    const ir = innerRadius;

    const x1 = center + r * Math.cos(startAngle);
    const y1 = center + r * Math.sin(startAngle);
    const x2 = center + r * Math.cos(endAngle);
    const y2 = center + r * Math.sin(endAngle);

    const ix1 = center + ir * Math.cos(endAngle);
    const iy1 = center + ir * Math.sin(endAngle);
    const ix2 = center + ir * Math.cos(startAngle);
    const iy2 = center + ir * Math.sin(startAngle);

    const largeArc = sliceAngle > Math.PI ? 1 : 0;

    const pathData = [
      `M ${x1} ${y1}`,
      `A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`,
      `L ${ix1} ${iy1}`,
      `A ${ir} ${ir} 0 ${largeArc} 0 ${ix2} ${iy2}`,
      'Z',
    ].join(' ');

    const percent = ((val / total) * 100).toFixed(1);

    return {
      label: d.label,
      value: val,
      percent,
      pathData,
      color: PALETTE[idx % PALETTE.length],
      idx,
    };
  });

  const activeSlice = hoveredIndex !== null ? slices[hoveredIndex] : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', gap: '16px' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center', gap: '24px' }}>
        {/* Donut SVG */}
        <div style={{ position: 'relative', width: size, height: size }}>
          <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ overflow: 'visible' }}>
            {slices.map((s) => (
              <path
                key={s.idx}
                d={s.pathData}
                fill={s.color}
                opacity={hoveredIndex !== null && hoveredIndex !== s.idx ? 0.6 : 0.95}
                stroke="#FFFFFF"
                strokeWidth="2"
                style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                onMouseEnter={() => setHoveredIndex(s.idx)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            ))}
          </svg>
          {/* Centered Donut Label */}
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
              pointerEvents: 'none',
              maxWidth: '90px',
            }}
          >
            {activeSlice ? (
              <>
                <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {activeSlice.label}
                </div>
                <div style={{ fontSize: '15px', fontWeight: 700, color: '#1E293B' }}>
                  {activeSlice.percent}%
                </div>
              </>
            ) : (
              <>
                <div style={{ fontSize: '10px', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total</div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>
                  {total >= 1000 ? `${(total / 1000).toFixed(1)}k` : total.toFixed(total < 10 ? 1 : 0)}
                  {unit}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Legend with Pills */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: '180px' }}>
          {slices.map((s) => (
            <div
              key={s.idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '4px 8px',
                borderRadius: '6px',
                background: hoveredIndex === s.idx ? '#F1F5F9' : 'transparent',
                cursor: 'pointer',
                transition: 'background 0.15s',
              }}
              onMouseEnter={() => setHoveredIndex(s.idx)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: s.color }} />
                <span style={{ fontSize: '12px', fontWeight: 500, color: '#334155' }}>{s.label}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B' }}>{s.value}{unit}</span>
                <span style={{ fontSize: '11px', color: '#94A3B8' }}>({s.percent}%)</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
