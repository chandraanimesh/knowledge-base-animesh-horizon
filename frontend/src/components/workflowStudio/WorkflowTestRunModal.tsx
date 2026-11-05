import React, { useState, useEffect } from 'react';
import type { WorkflowRunResponse } from '../../api/workflows.ts';
import { executeWorkflowTestRun } from '../../api/workflows.ts';

interface WorkflowTestRunModalProps {
  isOpen: boolean;
  onClose: () => void;
  workspaceId: string;
  workflowId: string;
  workflowName: string;
  versionId: string;
  versionNumber: number;
  versionStatus: string;
  triggerType?: string;
  token: string;
  onTestRunSuccess: (run: WorkflowRunResponse) => void;
}

export const WorkflowTestRunModal: React.FC<WorkflowTestRunModalProps> = ({
  isOpen,
  onClose,
  workspaceId,
  workflowId,
  workflowName,
  versionId,
  versionNumber,
  versionStatus,
  triggerType = 'MANUAL',
  token,
  onTestRunSuccess,
}) => {
  const getDefaultPayload = (trig: string): string => {
    switch (trig.toUpperCase()) {
      case 'WEBHOOK':
      case 'EVENT':
        return JSON.stringify(
          {
            event: 'company.created',
            company_id: 'comp-uuid-sample',
            name: 'Acme Corp',
            industry: 'Technology',
          },
          null,
          2
        );
      case 'SCHEDULE':
        return JSON.stringify(
          {
            scheduled_at: new Date().toISOString(),
            schedule_window: 'daily_summary',
            query: 'Daily automated knowledge briefing',
          },
          null,
          2
        );
      case 'MANUAL':
      default:
        return JSON.stringify(
          {
            query: 'Analyze quarterly customer acquisition performance and CRM pipeline data.',
            target_object: 'company',
            limit: 5,
          },
          null,
          2
        );
    }
  };

  const [payloadText, setPayloadText] = useState<string>(() => getDefaultPayload(triggerType));
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [maxTokens, setMaxTokens] = useState<number>(25000);
  const [maxProviderCalls, setMaxProviderCalls] = useState<number>(15);
  const [timeoutSeconds, setTimeoutSeconds] = useState<number>(180);
  const [showAdvancedLimits, setShowAdvancedLimits] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Re-sync sample payload when trigger type changes
  useEffect(() => {
    setPayloadText(getDefaultPayload(triggerType));
    setJsonError(null);
    setErrorMessage(null);
  }, [triggerType, versionId]);

  // Validate JSON on each keystroke
  const handlePayloadChange = (val: string) => {
    setPayloadText(val);
    try {
      JSON.parse(val);
      setJsonError(null);
    } catch (err: any) {
      setJsonError(err.message || 'Invalid JSON syntax');
    }
  };

  const handleFormatJson = () => {
    try {
      const parsed = JSON.parse(payloadText);
      setPayloadText(JSON.stringify(parsed, null, 2));
      setJsonError(null);
    } catch (err: any) {
      setJsonError(err.message || 'Cannot format invalid JSON');
    }
  };

  const handleStartTest = async () => {
    let parsedInputs: Record<string, any> = {};
    try {
      parsedInputs = JSON.parse(payloadText);
    } catch (err: any) {
      setJsonError('Please resolve JSON syntax errors before starting the test.');
      return;
    }

    try {
      setSubmitting(true);
      setErrorMessage(null);

      // Unique client-side idempotency key for this test run attempt
      const testIdempotencyKey = `studio-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

      const testRun = await executeWorkflowTestRun(
        workspaceId,
        workflowId,
        versionId,
        {
          inputs: parsedInputs,
          budget_limits: {
            max_tokens: maxTokens,
            max_provider_calls: maxProviderCalls,
            wall_clock_timeout_seconds: timeoutSeconds,
          },
          idempotency_key: testIdempotencyKey,
        },
        token
      );

      onTestRunSuccess(testRun);
      onClose();
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to execute test run.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-xs p-4 animate-in fade-in duration-150">
      <div className="bg-[#fcfcf9] border border-[#d4d4cc] rounded-xl shadow-2xl max-w-xl w-full flex flex-col max-h-[90vh] overflow-hidden text-slate-800">
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#d4d4cc] flex items-center justify-between bg-white shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-lg">🧪</span>
            <div>
              <h2 className="font-bold text-sm text-slate-800 flex items-center gap-2">
                <span>Test Run: {workflowName}</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase tracking-wider ${
                    versionStatus === 'ACTIVE'
                      ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                      : 'bg-amber-100 text-amber-800 border border-amber-300'
                  }`}
                >
                  v{versionNumber} ({versionStatus})
                </span>
              </h2>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Execute this draft workflow with a simulated trigger payload.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="text-slate-400 hover:text-slate-600 text-lg p-1 rounded hover:bg-slate-100 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 overflow-y-auto space-y-4 flex-1 text-xs">
          {/* Safe Mode Callout Banner */}
          <div className="p-3 bg-purple-50/80 border border-purple-200 rounded-lg text-purple-900 flex items-start gap-2.5">
            <span className="text-purple-600 text-base shrink-0 mt-0.5">🛡️</span>
            <div className="text-[11px] leading-relaxed">
              <span className="font-bold">Server-Controlled Test Isolation:</span> Real CRM
              mutations, email dispatch, and external writes are strictly simulated via dry-run
              adapters. Your draft version is executed safely without publishing or altering
              production schedules.
            </div>
          </div>

          {/* Trigger Info & Presets */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-700">Trigger Payload (JSON)</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                  {triggerType.toUpperCase()}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => handlePayloadChange(getDefaultPayload('MANUAL'))}
                  className="px-2 py-0.5 text-[10px] font-medium bg-slate-100 hover:bg-slate-200 text-slate-600 rounded border border-slate-200"
                >
                  Query Preset
                </button>
                <button
                  type="button"
                  onClick={() => handlePayloadChange(getDefaultPayload('WEBHOOK'))}
                  className="px-2 py-0.5 text-[10px] font-medium bg-slate-100 hover:bg-slate-200 text-slate-600 rounded border border-slate-200"
                >
                  CRM Event Preset
                </button>
                <button
                  type="button"
                  onClick={handleFormatJson}
                  className="px-2 py-0.5 text-[10px] font-medium bg-white hover:bg-slate-100 text-teal-700 rounded border border-teal-300"
                >
                  Format
                </button>
              </div>
            </div>

            {/* JSON Textarea */}
            <div className="relative">
              <textarea
                value={payloadText}
                onChange={(e) => handlePayloadChange(e.target.value)}
                rows={8}
                spellCheck={false}
                placeholder='{\n  "query": "Your sample question or test parameters"\n}'
                className={`w-full font-mono text-[11px] p-3 rounded-lg border bg-white focus:outline-none focus:ring-2 leading-relaxed ${
                  jsonError
                    ? 'border-red-400 focus:ring-red-200 bg-red-50/20'
                    : 'border-[#d4d4cc] focus:ring-teal-200 focus:border-teal-500'
                }`}
              />
              {jsonError && (
                <div className="mt-1 text-[11px] font-semibold text-red-600 flex items-center gap-1">
                  <span>✕</span>
                  <span>{jsonError}</span>
                </div>
              )}
            </div>
            <p className="text-[10px] text-slate-400 mt-1">
              Downstream DAG nodes can reference these fields via variables like{' '}
              <code className="text-teal-700 font-mono">{'{{trigger.query}}'}</code> or{' '}
              <code className="text-teal-700 font-mono">{'{{trigger.company_id}}'}</code>.
            </p>
          </div>

          {/* Advanced Budget Limits Collapsible */}
          <div className="border border-slate-200 rounded-lg overflow-hidden bg-slate-50/50">
            <button
              type="button"
              onClick={() => setShowAdvancedLimits(!showAdvancedLimits)}
              className="w-full px-3 py-2 text-left flex items-center justify-between text-slate-700 font-semibold text-[11px] hover:bg-slate-100 transition-colors"
            >
              <span className="flex items-center gap-1.5">
                <span>⚙️</span>
                <span>Execution Budget & Timeout Limits</span>
              </span>
              <span className="text-slate-400 text-xs">
                {showAdvancedLimits ? '▲ Hide' : '▼ Customize'}
              </span>
            </button>

            {showAdvancedLimits && (
              <div className="p-3 border-t border-slate-200 bg-white grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    min={1000}
                    max={200000}
                    step={1000}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(Number(e.target.value))}
                    className="w-full px-2 py-1 text-xs border border-slate-200 rounded font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1">
                    Max Calls
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={maxProviderCalls}
                    onChange={(e) => setMaxProviderCalls(Number(e.target.value))}
                    className="w-full px-2 py-1 text-xs border border-slate-200 rounded font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1">
                    Timeout (s)
                  </label>
                  <input
                    type="number"
                    min={10}
                    max={3600}
                    value={timeoutSeconds}
                    onChange={(e) => setTimeoutSeconds(Number(e.target.value))}
                    className="w-full px-2 py-1 text-xs border border-slate-200 rounded font-mono"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-[11px]">
              <span className="font-bold">Execution Failed: </span>
              {errorMessage}
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-5 py-3 border-t border-[#d4d4cc] flex items-center justify-end gap-2 bg-slate-50 shrink-0">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="px-4 py-1.5 text-xs font-semibold bg-white hover:bg-slate-100 text-slate-700 border border-[#d4d4cc] rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleStartTest}
            disabled={submitting || Boolean(jsonError)}
            className="px-4 py-1.5 text-xs font-semibold bg-teal-700 hover:bg-teal-800 text-white rounded-lg shadow-xs transition-colors flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting ? (
              <>
                <span className="animate-spin text-sm">⟳</span>
                <span>Executing Test DAG...</span>
              </>
            ) : (
              <>
                <span>▶</span>
                <span>Start Test Run</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

