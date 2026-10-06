import React, { useState } from 'react';
import { BarVisualization } from './types';

interface BarChartProps {
  spec: BarVisualization;
}

const PALETTE = [
  '#2563EB', // Blue
  '#10B981', // Emerald
  '#8B5CF6', // Purple
  '#F59E0B', // Amber
  '#EC4899', // Pink
  '#06B6D4', // Cyan
];

export const BarChart: React.FC<BarChartProps> = ({ spec }) => {
  const [hoveredPoint, setHoveredPoint] = useState<{
    seriesName: string;
    label: string;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  const series = spec.series || [];
  if (series.length === 0) return null;

  // Extract all distinct category labels in order
  const categories: string[] = [];
  series.forEach((s) => {
    s.data?.forEach((d) => {
      if (!categories.includes(d.label)) {
        categories.push(d.label);
      }
    });
  });

  if (categories.length === 0) return null;

  // Find max value across all series
  let maxValue = 0;
  series.forEach((s) => {
    s.data?.forEach((d) => {
      if (typeof d.value === 'number' && !isNaN(d.value) && d.value > maxValue) {
        maxValue = d.value;
      }
    });
  });
  if (maxValue <= 0) maxValue = 1;
  // Round max value up slightly for headroom
  const yUpper = Math.ceil(maxValue * 1.15 * 10) / 10;

  // Layout parameters
  const svgWidth = 640;
  const svgHeight = 280;
  const padding = { top: 20, right: 30, bottom: 45, left: 60 };
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = svgHeight - padding.top - padding.bottom;

  const numCategories = categories.length;
  const numSeries = series.length;
  const categoryWidth = chartWidth / numCategories;
  const groupPadding = categoryWidth * 0.2;
  const barWidth = Math.max(8, (categoryWidth - groupPadding) / numSeries);

  const unit = spec.y_axis?.unit ? ` ${spec.y_axis.unit}` : '';

  return (
    <div style={{ width: '100%', position: 'relative' }}>
      {/* Legend */}
      {numSeries > 1 && (
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', marginBottom: '12px', flexWrap: 'wrap' }}>
          {series.map((s, idx) => (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#475569' }}>
              <span
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '2px',
                  backgroundColor: PALETTE[idx % PALETTE.length],
                }}
              />
              <span style={{ fontWeight: 500 }}>{s.name}</span>
            </div>
          ))}
        </div>
      )}

      {/* SVG Canvas */}
      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: '100%', height: 'auto', maxHeight: '320px', display: 'block' }}
        >
          {/* Y Axis Gridlines & Labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const val = yUpper * (1 - ratio);
            const y = padding.top + chartHeight * ratio;
            return (
              <g key={ratio}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={padding.left + chartWidth}
                  y2={y}
                  stroke="#E2E8F0"
                  strokeDasharray={ratio === 1 ? 'none' : '3 3'}
                  strokeWidth="1"
                />
                <text
                  x={padding.left - 8}
                  y={y + 4}
                  textAnchor="end"
                  fontSize="11"
                  fill="#94A3B8"
                  fontFamily="sans-serif"
                >
                  {val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toFixed(val < 10 ? 1 : 0)}
                  {ratio === 0 && unit ? unit : ''}
                </text>
              </g>
            );
          })}

          {/* Bars */}
          {categories.map((cat, catIdx) => {
            const groupX = padding.left + catIdx * categoryWidth + groupPadding / 2;
            return (
              <g key={catIdx}>
                {series.map((s, sIdx) => {
                  const pt = s.data?.find((d) => d.label === cat);
                  const val = pt ? pt.value : 0;
                  const barH = Math.max(0, (val / yUpper) * chartHeight);
                  const barX = groupX + sIdx * barWidth;
                  const barY = padding.top + chartHeight - barH;
                  const color = PALETTE[sIdx % PALETTE.length];

                  return (
                    <g key={sIdx}>
                      <rect
                        x={barX}
                        y={barY}
                        width={barWidth - 2}
                        height={barH}
                        rx="3"
                        ry="3"
                        fill={color}
                        opacity={hoveredPoint && (hoveredPoint.seriesName !== s.name || hoveredPoint.label !== cat) ? 0.6 : 0.95}
                        style={{ cursor: 'pointer', transition: 'opacity 0.15s' }}
                        onMouseEnter={(e) => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          setHoveredPoint({
                            seriesName: s.name,
                            label: cat,
                            value: val,
                            x: barX + (barWidth - 2) / 2,
                            y: barY - 8,
                          });
                        }}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                      {/* Bar top value label */}
                      {barH > 18 && (
                        <text
                          x={barX + (barWidth - 2) / 2}
                          y={barY - 4}
                          textAnchor="middle"
                          fontSize="9.5"
                          fontWeight="600"
                          fill="#475569"
                          fontFamily="sans-serif"
                        >
                          {val}
                        </text>
                      )}
                    </g>
                  );
                })}

                {/* X Category Label */}
                <text
                  x={padding.left + catIdx * categoryWidth + categoryWidth / 2}
                  y={padding.top + chartHeight + 18}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight="500"
                  fill="#475569"
                  fontFamily="sans-serif"
                >
                  {cat.length > 14 ? `${cat.substring(0, 12)}…` : cat}
                </text>
              </g>
            );
          })}

          {/* X Axis Label */}
          {spec.x_axis?.label && (
            <text
              x={padding.left + chartWidth / 2}
              y={svgHeight - 4}
              textAnchor="middle"
              fontSize="11"
              fill="#64748B"
              fontFamily="sans-serif"
            >
              {spec.x_axis.label}
            </text>
          )}
        </svg>
      </div>

      {/* Floating Hover Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: 'absolute',
            left: `${(hoveredPoint.x / svgWidth) * 100}%`,
            top: `${(hoveredPoint.y / svgHeight) * 100}%`,
            transform: 'translate(-50%, -100%)',
            background: 'rgba(15, 23, 42, 0.92)',
            color: '#FFFFFF',
            padding: '4px 8px',
            borderRadius: '6px',
            fontSize: '11px',
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            zIndex: 10,
          }}
        >
          <div style={{ fontWeight: 600 }}>{hoveredPoint.label}</div>
          <div style={{ opacity: 0.9 }}>
            {hoveredPoint.seriesName}: <b>{hoveredPoint.value}</b>
            {unit}
          </div>
        </div>
      )}
    </div>
  );
};
