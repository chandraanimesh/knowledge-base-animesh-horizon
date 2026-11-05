import React, { useState } from 'react';
import { ACTION_CATALOG, TRIGGER_CATALOG } from './types';
import { WorkflowActionType, WorkflowTriggerType } from '../../api/workflowDefinitions';

interface NodeLibrarySidebarProps {
  onAddNode: (type: WorkflowActionType) => void;
  onChangeTrigger: (type: WorkflowTriggerType) => void;
  activeTriggerType: WorkflowTriggerType;
  isReadOnly?: boolean;
}

export const NodeLibrarySidebar: React.FC<NodeLibrarySidebarProps> = ({
  onAddNode,
  onChangeTrigger,
  activeTriggerType,
  isReadOnly = false,
}) => {
  const [searchFilter, setSearchFilter] = useState('');
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({
    triggers: false,
    knowledge: false,
    control: false,
    actions: false,
  });

  const toggleCategory = (cat: string) => {
    setCollapsedCategories((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  const onDragStart = (event: React.DragEvent, nodeType: WorkflowActionType) => {
    if (isReadOnly) return;
    event.dataTransfer.setData('application/reactflow/type', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const knowledgeActions = Object.values(ACTION_CATALOG).filter(
    (a) =>
      a.category === 'knowledge' &&
      (a.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
        a.description.toLowerCase().includes(searchFilter.toLowerCase()))
  );

  const controlActions = Object.values(ACTION_CATALOG).filter(
    (a) =>
      a.category === 'control' &&
      (a.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
        a.description.toLowerCase().includes(searchFilter.toLowerCase()))
  );

  const crmActions = Object.values(ACTION_CATALOG).filter(
    (a) =>
      a.category === 'actions' &&
      (a.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
        a.description.toLowerCase().includes(searchFilter.toLowerCase()))
  );

  const triggers = Object.values(TRIGGER_CATALOG).filter(
    (t) =>
      t.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
      t.description.toLowerCase().includes(searchFilter.toLowerCase())
  );

  return (
    <aside className="w-72 bg-[#fafaf7] border-r border-[#d4d4cc] flex flex-col h-full select-none shrink-0 min-h-0">
      {/* Header & Search */}
      <div className="p-3 border-b border-[#d4d4cc] bg-[#f0f0eb]">
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-2">
          Node Library
        </h2>
        <input
          type="text"
          placeholder="Search components..."
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          className="w-full text-xs px-2.5 py-1.5 rounded-md border border-[#d4d4cc] bg-white focus:outline-none focus:ring-1 focus:ring-teal-600"
        />
        {isReadOnly && (
          <p className="mt-2 text-[10px] text-amber-700 bg-amber-50 p-1.5 rounded border border-amber-200">
            🔒 View-only mode. Editor role required to add nodes.
          </p>
        )}
      </div>

      {/* Accordion Categories */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4 min-h-0 overscroll-contain">
        {/* Category 1: Triggers */}
        <div>
          <button
            onClick={() => toggleCategory('triggers')}
            className="flex items-center justify-between w-full text-left text-[11px] font-bold text-slate-600 uppercase tracking-wider mb-1.5 hover:text-slate-900"
          >
            <span>Triggers ({triggers.length})</span>
            <span className="text-slate-400">{collapsedCategories.triggers ? '▶' : '▼'}</span>
          </button>

          {!collapsedCategories.triggers && (
            <div className="space-y-1.5 pl-1">
              {triggers.map((trig) => {
                const isActive = activeTriggerType === trig.type;
                return (
                  <div
                    key={trig.type}
                    onClick={() => !isReadOnly && onChangeTrigger(trig.type)}
                    className={`p-2 rounded-lg border text-left transition-all ${
                      isActive
                        ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-300'
                        : 'border-slate-200 bg-white hover:border-blue-300'
                    } ${isReadOnly ? 'cursor-not-allowed opacity-75' : 'cursor-pointer'}`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-800">{trig.name}</span>
                      {isActive && (
                        <span className="text-[10px] font-bold text-blue-600 uppercase">
                          Active
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-0.5 line-clamp-2">
                      {trig.description}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Category 2: Knowledge Actions */}
        <div>
          <button
            onClick={() => toggleCategory('knowledge')}
            className="flex items-center justify-between w-full text-left text-[11px] font-bold text-emerald-800 uppercase tracking-wider mb-1.5 hover:text-emerald-950"
          >
            <span>Knowledge & RAG ({knowledgeActions.length})</span>
            <span className="text-emerald-600">{collapsedCategories.knowledge ? '▶' : '▼'}</span>
          </button>

          {!collapsedCategories.knowledge && (
            <div className="space-y-1.5 pl-1">
              {knowledgeActions.map((action) => (
                <div
                  key={action.type}
                  draggable={!isReadOnly}
                  onDragStart={(e) => onDragStart(e, action.type)}
                  onClick={() => !isReadOnly && onAddNode(action.type)}
                  className={`p-2.5 rounded-lg border bg-white shadow-xs transition-all ${
                    isReadOnly
                      ? 'cursor-not-allowed opacity-70 border-slate-200'
                      : 'cursor-grab hover:shadow-sm hover:border-emerald-400 active:cursor-grabbing border-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span
                      className="text-[9px] font-bold px-1.5 py-0.5 rounded uppercase"
                      style={{
                        backgroundColor: action.color.badgeBg,
                        color: action.color.text,
                      }}
                    >
                      {action.type}
                    </span>
                    <button
                      type="button"
                      disabled={isReadOnly}
                      className="text-[10px] text-emerald-700 font-semibold hover:underline"
                    >
                      + Add
                    </button>
                  </div>
                  <div className="text-xs font-semibold text-slate-800">{action.name}</div>
                  <p className="text-[10px] text-slate-500 mt-0.5 line-clamp-2">
                    {action.description}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Category 3: Control Flow */}
        <div>
          <button
            onClick={() => toggleCategory('control')}
            className="flex items-center justify-between w-full text-left text-[11px] font-bold text-amber-800 uppercase tracking-wider mb-1.5 hover:text-amber-950"
          >
            <span>Control Flow ({controlActions.length})</span>
            <span className="text-amber-600">{collapsedCategories.control ? '▶' : '▼'}</span>
          </button>

          {!collapsedCategories.control && (
            <div className="space-y-1.5 pl-1">
              {controlActions.map((action) => (
                <div
                  key={action.type}
                  draggable={!isReadOnly}
                  onDragStart={(e) => onDragStart(e, action.type)}
                  onClick={() => !isReadOnly && onAddNode(action.type)}
                  className={`p-2.5 rounded-lg border bg-white shadow-xs transition-all ${
                    isReadOnly
                      ? 'cursor-not-allowed opacity-70 border-slate-200'
                      : 'cursor-grab hover:shadow-sm hover:border-amber-400 active:cursor-grabbing border-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span
                      className="text-[9px] font-bold px-1.5 py-0.5 rounded uppercase"
                      style={{
                        backgroundColor: action.color.badgeBg,
                        color: action.color.text,
                      }}
                    >
                      {action.type}
                    </span>
                    <button
                      type="button"
                      disabled={isReadOnly}
                      className="text-[10px] text-amber-700 font-semibold hover:underline"
                    >
                      + Add
                    </button>
                  </div>
                  <div className="text-xs font-semibold text-slate-800">{action.name}</div>
                  <p className="text-[10px] text-slate-500 mt-0.5 line-clamp-2">
                    {action.description}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Category 4: Restricted Actions */}
        <div>
          <button
            onClick={() => toggleCategory('actions')}
            className="flex items-center justify-between w-full text-left text-[11px] font-bold text-purple-800 uppercase tracking-wider mb-1.5 hover:text-purple-950"
          >
            <span>Restricted Actions ({crmActions.length})</span>
            <span className="text-purple-600">{collapsedCategories.actions ? '▶' : '▼'}</span>
          </button>

          {!collapsedCategories.actions && (
            <div className="space-y-1.5 pl-1">
              {crmActions.map((action) => (
                <div
                  key={action.type}
                  draggable={!isReadOnly}
                  onDragStart={(e) => onDragStart(e, action.type)}
                  onClick={() => !isReadOnly && onAddNode(action.type)}
                  className={`p-2.5 rounded-lg border bg-white shadow-xs transition-all ${
                    isReadOnly
                      ? 'cursor-not-allowed opacity-70 border-slate-200'
                      : 'cursor-grab hover:shadow-sm hover:border-purple-400 active:cursor-grabbing border-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span
                      className="text-[9px] font-bold px-1.5 py-0.5 rounded uppercase"
                      style={{
                        backgroundColor: action.color.badgeBg,
                        color: action.color.text,
                      }}
                    >
                      {action.type}
                    </span>
                    <button
                      type="button"
                      disabled={isReadOnly}
                      className="text-[10px] text-purple-700 font-semibold hover:underline"
                    >
                      + Add
                    </button>
                  </div>
                  <div className="text-xs font-semibold text-slate-800">{action.name}</div>
                  <p className="text-[10px] text-slate-500 mt-0.5 line-clamp-2">
                    {action.description}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

