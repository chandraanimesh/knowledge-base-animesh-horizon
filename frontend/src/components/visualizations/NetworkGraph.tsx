import React, { useState } from 'react';
import { NetworkVisualization, NetworkNodeType } from './types';

interface NetworkGraphProps {
  spec: NetworkVisualization;
}

const NODE_COLORS: Record<NetworkNodeType, { bg: string; border: string; text: string; icon: string }> = {
  company: { bg: '#EFF6FF', border: '#3B82F6', text: '#1D4ED8', icon: '🏢' },
  person: { bg: '#F0FDF4', border: '#10B981', text: '#047857', icon: '👤' },
  department: { bg: '#FDF4FF', border: '#D946EF', text: '#A21CAF', icon: '🏛️' },
  document: { bg: '#F8FAFC', border: '#64748B', text: '#334155', icon: '📄' },
  topic: { bg: '#FEF3C7', border: '#F59E0B', text: '#B45309', icon: '💡' },
  deal: { bg: '#ECFDF5', border: '#059669', text: '#065F46', icon: '💼' },
  event: { bg: '#FFF1F2', border: '#F43F5E', text: '#BE123C', icon: '📅' },
  other: { bg: '#F1F5F9', border: '#94A3B8', text: '#475569', icon: '🔹' },
};

export const NetworkGraph: React.FC<NetworkGraphProps> = ({ spec }) => {
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  const nodes = spec.nodes || [];
  const edges = spec.edges || [];

  if (nodes.length === 0) return null;

  const svgWidth = 600;
  const svgHeight = 320;
  const centerX = svgWidth / 2;
  const centerY = svgHeight / 2;
  const radiusX = 220;
  const radiusY = 110;

  // Calculate node positions in an elliptical / structured layout
  const nodePositions: Record<string, { x: number; y: number }> = {};
  const numNodes = nodes.length;

  if (numNodes === 1) {
    nodePositions[nodes[0].id] = { x: centerX, y: centerY };
  } else if (numNodes === 2) {
    nodePositions[nodes[0].id] = { x: centerX - 140, y: centerY };
    nodePositions[nodes[1].id] = { x: centerX + 140, y: centerY };
  } else if (numNodes === 3) {
    nodePositions[nodes[0].id] = { x: centerX, y: centerY - 90 };
    nodePositions[nodes[1].id] = { x: centerX - 160, y: centerY + 70 };
    nodePositions[nodes[2].id] = { x: centerX + 160, y: centerY + 70 };
  } else {
    nodes.forEach((n, idx) => {
      const angle = (idx / numNodes) * 2 * Math.PI - Math.PI / 2;
      nodePositions[n.id] = {
        x: centerX + radiusX * Math.cos(angle),
        y: centerY + radiusY * Math.sin(angle),
      };
    });
  }

  // Find connected edges for hover highlighting
  const connectedNodeIds = new Set<string>();
  if (hoveredNodeId) {
    connectedNodeIds.add(hoveredNodeId);
    edges.forEach((e) => {
      if (e.source === hoveredNodeId) connectedNodeIds.add(e.target);
      if (e.target === hoveredNodeId) connectedNodeIds.add(e.source);
    });
  }

  return (
    <div style={{ width: '100%', position: 'relative' }}>
      {/* Type Badges / Legend */}
      <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginBottom: '10px', flexWrap: 'wrap' }}>
        {Array.from(new Set(nodes.map((n) => n.type || 'other'))).map((t) => {
          const style = NODE_COLORS[t as NetworkNodeType] || NODE_COLORS.other;
          return (
            <span
              key={t}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '11px',
                padding: '2px 8px',
                borderRadius: '12px',
                backgroundColor: style.bg,
                color: style.text,
                border: `1px solid ${style.border}`,
                textTransform: 'capitalize',
              }}
            >
              <span>{style.icon}</span>
              <span>{t}</span>
            </span>
          );
        })}
      </div>

      {/* SVG Canvas */}
      <div style={{ width: '100%', overflowX: 'auto', background: '#FAFAFA', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: '100%', height: 'auto', maxHeight: '340px', display: 'block' }}
        >
          <defs>
            {/* Directional Arrow Marker */}
            <marker
              id="network-arrow"
              viewBox="0 0 10 10"
              refX="22"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#94A3B8" />
            </marker>
            <marker
              id="network-arrow-active"
              viewBox="0 0 10 10"
              refX="22"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB" />
            </marker>
          </defs>

          {/* Edges */}
          {edges.map((e, eIdx) => {
            const src = nodePositions[e.source];
            const tgt = nodePositions[e.target];
            if (!src || !tgt) return null;

            const isEdgeActive =
              hoveredNodeId && (e.source === hoveredNodeId || e.target === hoveredNodeId);
            const isDimmed = hoveredNodeId && !isEdgeActive;

            const midX = (src.x + tgt.x) / 2;
            const midY = (src.y + tgt.y) / 2;

            return (
              <g key={eIdx} opacity={isDimmed ? 0.25 : 1} style={{ transition: 'opacity 0.2s' }}>
                <line
                  x1={src.x}
                  y1={src.y}
                  x2={tgt.x}
                  y2={tgt.y}
                  stroke={isEdgeActive ? '#2563EB' : '#CBD5E1'}
                  strokeWidth={isEdgeActive ? 2.5 : 1.5}
                  markerEnd={isEdgeActive ? 'url(#network-arrow-active)' : 'url(#network-arrow)'}
                />
                {/* Edge Label Badge */}
                {e.label && (
                  <g transform={`translate(${midX}, ${midY})`}>
                    <rect
                      x={-(e.label.length * 3.2 + 6)}
                      y="-9"
                      width={e.label.length * 6.4 + 12}
                      height="18"
                      rx="4"
                      fill="#FFFFFF"
                      stroke={isEdgeActive ? '#93C5FD' : '#E2E8F0'}
                      strokeWidth="1"
                    />
                    <text
                      x="0"
                      y="3.5"
                      textAnchor="middle"
                      fontSize="9.5"
                      fontWeight="500"
                      fill={isEdgeActive ? '#1D4ED8' : '#64748B'}
                      fontFamily="sans-serif"
                    >
                      {e.label}
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {nodes.map((n) => {
            const pos = nodePositions[n.id];
            if (!pos) return null;

            const style = NODE_COLORS[n.type || 'other'] || NODE_COLORS.other;
            const isHovered = hoveredNodeId === n.id;
            const isConnected = connectedNodeIds.has(n.id);
            const isDimmed = hoveredNodeId && !isConnected;

            const nodeWidth = Math.max(90, n.label.length * 7.5 + 28);
            const nodeHeight = 32;

            return (
              <g
                key={n.id}
                transform={`translate(${pos.x}, ${pos.y})`}
                opacity={isDimmed ? 0.3 : 1}
                style={{ cursor: 'pointer', transition: 'all 0.2s ease' }}
                onMouseEnter={() => setHoveredNodeId(n.id)}
                onMouseLeave={() => setHoveredNodeId(null)}
              >
                {/* Node Pill */}
                <rect
                  x={-nodeWidth / 2}
                  y={-nodeHeight / 2}
                  width={nodeWidth}
                  height={nodeHeight}
                  rx="16"
                  ry="16"
                  fill={style.bg}
                  stroke={isHovered ? '#2563EB' : style.border}
                  strokeWidth={isHovered ? 2.5 : 1.5}
                  filter="drop-shadow(0 1px 2px rgba(0, 0, 0, 0.05))"
                />

                {/* Node Icon & Label */}
                <text
                  x={-nodeWidth / 2 + 14}
                  y="4"
                  fontSize="12"
                  textAnchor="middle"
                >
                  {style.icon}
                </text>
                <text
                  x={8}
                  y="4"
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight="600"
                  fill={style.text}
                  fontFamily="sans-serif"
                >
                  {n.label.length > 18 ? `${n.label.substring(0, 16)}…` : n.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
