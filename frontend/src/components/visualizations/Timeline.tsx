import React from 'react';
import { TimelineVisualization } from './types';

interface TimelineProps {
  spec: TimelineVisualization;
}

export const Timeline: React.FC<TimelineProps> = ({ spec }) => {
  const events = spec.events || [];
  if (events.length === 0) return null;

  return (
    <div style={{ width: '100%', padding: '8px 0' }}>
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Continuous Spine Line */}
        <div
          style={{
            position: 'absolute',
            left: '15px',
            top: '12px',
            bottom: '12px',
            width: '2px',
            backgroundColor: '#E2E8F0',
            zIndex: 0,
          }}
        />

        {events.map((evt, idx) => (
          <div key={idx} style={{ position: 'relative', display: 'flex', alignItems: 'flex-start', gap: '16px', zIndex: 1 }}>
            {/* Timeline Dot Indicator */}
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                backgroundColor: '#EFF6FF',
                border: '2px solid #3B82F6',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                color: '#1D4ED8',
                fontSize: '11px',
                fontWeight: 700,
                boxShadow: '0 1px 3px rgba(59, 130, 246, 0.2)',
              }}
            >
              {idx + 1}
            </div>

            {/* Event Card */}
            <div
              style={{
                flex: 1,
                background: '#FFFFFF',
                borderRadius: '8px',
                border: '1px solid #E2E8F0',
                padding: '10px 14px',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#1E293B' }}>{evt.label}</span>
                <span
                  style={{
                    fontSize: '10.5px',
                    fontFamily: 'monospace',
                    fontWeight: 600,
                    color: '#2563EB',
                    backgroundColor: '#EFF6FF',
                    padding: '2px 8px',
                    borderRadius: '10px',
                    border: '1px solid #DBEAFE',
                  }}
                >
                  📅 {evt.date}
                </span>
              </div>
              <p style={{ margin: 0, fontSize: '12px', color: '#475569', lineHeight: 1.5 }}>{evt.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
