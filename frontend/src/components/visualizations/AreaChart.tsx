import React, { useState } from 'react';
import { AreaVisualization } from './types';

interface AreaChartProps {
  spec: AreaVisualization;
}

const PALETTE = [
  '#2563EB', // Blue
  '#10B981', // Emerald
  '#8B5CF6', // Purple
  '#F59E0B', // Amber
  '#EC4899', // Pink
];

export const AreaChart: React.FC<AreaChartProps> = ({ spec }) => {
  const [hoveredPoint, setHoveredPoint] = useState<{
    seriesName: string;
    label: string;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  const series = spec.series || [];
  if (series.length === 0) return null;

  const labels: string[] = [];
  series.forEach((s) => {
    s.data?.forEach((d) => {
      if (!labels.includes(d.label)) {
        labels.push(d.label);
      }
    });
  });

  if (labels.length === 0) return null;

  let maxValue = 0;
  series.forEach((s) => {
    s.data?.forEach((d) => {
      if (typeof d.value === 'number' && !isNaN(d.value) && d.value > maxValue) {
        maxValue = d.value;
      }
    });
  });
  if (maxValue <= 0) maxValue = 1;
  const yUpper = Math.ceil(maxValue * 1.15 * 10) / 10;

  const svgWidth = 640;
  const svgHeight = 280;
  const padding = { top: 20, right: 30, bottom: 45, left: 60 };
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = svgHeight - padding.top - padding.bottom;

  const numPoints = labels.length;
  const xStep = numPoints > 1 ? chartWidth / (numPoints - 1) : chartWidth / 2;

  const getX = (idx: number) => padding.left + (numPoints > 1 ? idx * xStep : chartWidth / 2);
  const getY = (val: number) => padding.top + chartHeight - (val / yUpper) * chartHeight;
  const bottomY = padding.top + chartHeight;

  const unit = spec.y_axis?.unit ? ` ${spec.y_axis.unit}` : '';

  return (
    <div style={{ width: '100%', position: 'relative' }}>
      {/* Legend */}
      {series.length > 1 && (
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', marginBottom: '12px', flexWrap: 'wrap' }}>
          {series.map((s, idx) => (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#475569' }}>
              <span
                style={{
                  width: '12px',
                  height: '3px',
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
          <defs>
            {series.map((_, sIdx) => {
              const color = PALETTE[sIdx % PALETTE.length];
              return (
                <linearGradient key={sIdx} id={`area-grad-${sIdx}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity="0.35" />
                  <stop offset="100%" stopColor={color} stopOpacity="0.03" />
                </linearGradient>
              );
            })}
          </defs>

          {/* Y Axis Gridlines */}
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

          {/* Area Fills and Lines */}
          {series.map((s, sIdx) => {
            const color = PALETTE[sIdx % PALETTE.length];
            const points: { x: number; y: number; val: number; label: string }[] = [];

            labels.forEach((l, lIdx) => {
              const pt = s.data?.find((d) => d.label === l);
              if (pt && typeof pt.value === 'number') {
                points.push({
                  x: getX(lIdx),
                  y: getY(pt.value),
                  val: pt.value,
                  label: l,
                });
              }
            });

            if (points.length === 0) return null;

            const linePath = points.reduce((acc, curr, i) => {
              return i === 0 ? `M ${curr.x} ${curr.y}` : `${acc} L ${curr.x} ${curr.y}`;
            }, '');

            const areaPath = `${linePath} L ${points[points.length - 1].x} ${bottomY} L ${points[0].x} ${bottomY} Z`;

            return (
              <g key={sIdx}>
                {/* Area Gradient Fill */}
                <path d={areaPath} fill={`url(#area-grad-${sIdx})`} />

                {/* Top Line */}
                <path
                  d={linePath}
                  fill="none"
                  stroke={color}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Points */}
                {points.map((p, pIdx) => (
                  <circle
                    key={pIdx}
                    cx={p.x}
                    cy={p.y}
                    r={hoveredPoint && hoveredPoint.seriesName === s.name && hoveredPoint.label === p.label ? 6 : 3.5}
                    fill="#FFFFFF"
                    stroke={color}
                    strokeWidth="2.5"
                    style={{ cursor: 'pointer', transition: 'r 0.15s' }}
                    onMouseEnter={() => {
                      setHoveredPoint({
                        seriesName: s.name,
                        label: p.label,
                        value: p.val,
                        x: p.x,
                        y: p.y - 8,
                      });
                    }}
                    onMouseLeave={() => setHoveredPoint(null)}
                  />
                ))}
              </g>
            );
          })}

          {/* X Axis Labels */}
          {labels.map((l, lIdx) => (
            <text
              key={lIdx}
              x={getX(lIdx)}
              y={padding.top + chartHeight + 18}
              textAnchor="middle"
              fontSize="11"
              fontWeight="500"
              fill="#475569"
              fontFamily="sans-serif"
            >
              {l.length > 12 ? `${l.substring(0, 10)}…` : l}
            </text>
          ))}

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
