import React, { useState } from 'react';
import { VisualizationSpec } from './types';
import { BarChart } from './BarChart';
import { LineChart } from './LineChart';
import { PieChart } from './PieChart';
import { AreaChart } from './AreaChart';
import { DataTable } from './DataTable';
import { NetworkGraph } from './NetworkGraph';
import { Timeline } from './Timeline';

export interface VisualizationRendererProps {
  spec?: VisualizationSpec | null;
}

const TYPE_ICONS: Record<string, string> = {
  bar: '📊',
  line: '📈',
  pie: '🍩',
  area: '📉',
  table: '📋',
  network: '🕸️',
  timeline: '⏳',
};

const TYPE_LABELS: Record<string, string> = {
  bar: 'Bar Chart',
  line: 'Line Chart',
  pie: 'Pie Chart',
  area: 'Area Chart',
  table: 'Data Table',
  network: 'Network Graph',
  timeline: 'Timeline',
};

export const VisualizationRenderer: React.FC<VisualizationRendererProps> = ({ spec }) => {
  const [showRawJson, setShowRawJson] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!spec || !spec.type || !spec.title) {
    return null;
  }

  const icon = TYPE_ICONS[spec.type] || '📊';
  const typeLabel = TYPE_LABELS[spec.type] || spec.type.toUpperCase();

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(spec, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderContent = () => {
    switch (spec.type) {
      case 'bar':
        return <BarChart spec={spec} />;
      case 'line':
        return <LineChart spec={spec} />;
      case 'pie':
        return <PieChart spec={spec} />;
      case 'area':
        return <AreaChart spec={spec} />;
      case 'table':
        return <DataTable spec={spec} />;
      case 'network':
        return <NetworkGraph spec={spec} />;
      case 'timeline':
        return <Timeline spec={spec} />;
      default:
        return null;
    }
  };

  return (
    <div
      style={{
        margin: '14px 0',
        padding: '14px 18px',
        borderRadius: '12px',
        backgroundColor: '#FFFFFF',
        border: '1px solid #E2E8F0',
        boxShadow: '0 2px 6px rgba(0, 0, 0, 0.04)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        maxWidth: '100%',
        boxSizing: 'border-box',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
          borderBottom: '1px solid #F1F5F9',
          paddingBottom: '10px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '16px' }}>{icon}</span>
          <span style={{ fontSize: '13.5px', fontWeight: 600, color: '#1E293B' }}>{spec.title}</span>
          <span
            style={{
              fontSize: '10px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              padding: '2px 6px',
              borderRadius: '6px',
              backgroundColor: '#EFF6FF',
              color: '#2563EB',
              border: '1px solid #DBEAFE',
            }}
          >
            {typeLabel}
          </span>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            type="button"
            onClick={() => setShowRawJson(!showRawJson)}
            style={{
              fontSize: '11px',
              color: '#64748B',
              background: '#F8FAFC',
              border: '1px solid #E2E8F0',
              borderRadius: '6px',
              padding: '3px 8px',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
          >
            {showRawJson ? 'View Chart' : '{ } Spec'}
          </button>
          <button
            type="button"
            onClick={handleCopy}
            title="Copy visualization JSON"
            style={{
              fontSize: '11px',
              color: copied ? '#10B981' : '#64748B',
              background: '#F8FAFC',
              border: '1px solid #E2E8F0',
              borderRadius: '6px',
              padding: '3px 8px',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
          >
            {copied ? '✓ Copied' : 'Copy'}
          </button>
        </div>
      </div>

      {/* Body */}
      {showRawJson ? (
        <pre
          style={{
            margin: 0,
            padding: '10px 12px',
            backgroundColor: '#0F172A',
            color: '#F8FAFC',
            borderRadius: '8px',
            fontSize: '11px',
            overflowX: 'auto',
            maxHeight: '260px',
          }}
        >
          {JSON.stringify(spec, null, 2)}
        </pre>
      ) : (
        <div style={{ width: '100%' }}>{renderContent()}</div>
      )}
    </div>
  );
};
