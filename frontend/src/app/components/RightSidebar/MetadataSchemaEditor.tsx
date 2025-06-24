"use client";

import { UIMetadataField } from "@/types/collections";
import { useState } from "react";

type FieldType = "string" | "datetime";

interface Props {
  schema: UIMetadataField[];
  setSchema: (fields: UIMetadataField[]) => void;
  allowNewField?: boolean;
}

export default function MetadataSchemaEditor({
  schema,
  setSchema,
  allowNewField = true,
}: Props) {
  const [showSchemaEditor, setShowSchemaEditor] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  // Store edit values by index
  const [editValues, setEditValues] = useState<Record<number, UIMetadataField>>({});
  const [newField, setNewField] = useState<UIMetadataField>({
    name: "",
    type: "string",
    optional: true,
  });

  const startEditing = (idx: number) => {
    setEditingIndex(idx);
    setEditValues((prev) => ({
      ...prev,
      [idx]: {
        name: schema[idx].name,
        type: schema[idx].type,
        optional: schema[idx].optional ?? true,
      },
    }));
  };

  const updateEditValue = (idx: number, updatedValue: Partial<UIMetadataField>) => {
    setEditValues((prev) => ({
      ...prev,
      [idx]: {
        ...prev[idx],
        ...updatedValue,
      },
    }));
  };

  const commitUpdate = (index: number) => {
    const editValue = editValues[index];
    if (!editValue) return;
    const updated = [...schema];
    updated[index] = {
      ...editValue,
      name: editValue.name.trim(),
    };
    setSchema(updated);
    setEditingIndex(null);
    setEditValues((prev) => {
      const copy = { ...prev };
      delete copy[index];
      return copy;
    });
  };

  const handleAdd = () => {
    if (!newField.name.trim()) return;
    const updated = [...schema, { ...newField, name: newField.name.trim() }];
    setSchema(updated);
    setNewField({ name: "", type: "string", optional: true });
  };

  const handleDelete = (index: number) => {
    const updated = schema.filter((_, i) => i !== index);
    setSchema(updated);
    setEditingIndex(null);
    setEditValues((prev) => {
      const copy = { ...prev };
      delete copy[index];
      return copy;
    });
  };

  if (!allowNewField && schema.length === 0) return null;

  return (
    <div className="rounded border border-neutral-700 bg-neutral-900 text-white mt-4 my-3">
      <button
        type="button"
        onClick={() => setShowSchemaEditor((prev) => !prev)}
        className="w-full flex justify-between items-center px-4 py-3 text-sm font-medium hover:bg-neutral-800 transition-colors"
      >
        <span>Metadata Schema</span>
        <span>{showSchemaEditor ? "-" : "+"}</span>
      </button>

      {showSchemaEditor && (
        <div className="border-t border-neutral-800 p-4">
          <p className="text-sm text-neutral-300 mb-3">
            Define metadata fields for this collection.
          </p>

          <div className="space-y-3">
            {schema.map((field, idx) =>
              editingIndex === idx ? (
                <div
                  key={idx}
                  className="rounded-lg border border-neutral-700 bg-neutral-800 p-4 space-y-3"
                >
                  <div className="grid grid-cols-3 gap-4 items-end">
                    <div>
                      <label className="text-xs block mb-1 text-neutral-400">Field</label>
                      <input
                        autoFocus
                        value={editValues[idx]?.name || ""}
                        onChange={(e) =>
                          updateEditValue(idx, { name: e.target.value })
                        }
                        onKeyDown={(e) => e.key === "Enter" && commitUpdate(idx)}
                        className="w-full h-7 bg-neutral-900 px-3 py-1 rounded"
                      />
                    </div>
                    <div>
                      <label className="text-xs block mb-1 text-neutral-400">Type</label>
                      <select
                        value={editValues[idx]?.type || "string"}
                        onChange={(e) =>
                          updateEditValue(idx, { type: e.target.value as FieldType })
                        }
                        className="bg-neutral-900 px-3 py-1 rounded w-full"
                      >
                        <option value="string">string</option>
                        <option value="datetime">datetime</option>
                      </select>
                    </div>
                  </div>
                  <div className="flex justify-end items-center mt-2 space-x-2">
                    <button onClick={() => setEditingIndex(null)} title="Cancel">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>

                    <button onClick={() => commitUpdate(idx)} title="Save">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              ) : (
                <div key={idx} className="rounded-lg border border-neutral-700 bg-neutral-800 p-2">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="text-sm font-semibold text-white">{field.name}</div>
                      <div className="text-xs text-neutral-400 mt-1">
                        Type: <span className="font-medium">{field.type}</span>
                      </div>
                    </div>
                    {allowNewField && (
                      <div className="space-x-3 mt-1">
                        <button
                          onClick={() => startEditing(idx)}
                          title="Edit"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M12 20h9" />
                            <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4Z" />
                          </svg>
                        </button>
                        <button onClick={() => handleDelete(idx)} title="Delete">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M3 6h18" />
                            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                          </svg>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )
            )}
          </div>

          {allowNewField && (
            <div className="border-t border-neutral-700 pt-5 mt-6">
              <h4 className="text-sm font-medium mb-3">Add New Field</h4>
              <div className="grid grid-cols-[1fr_1fr_auto] gap-3 items-end">
                <div>
                  <label className="text-xs block mb-1 text-neutral-400">Field name</label>
                  <input
                    placeholder="e.g. category"
                    value={newField.name}
                    onChange={(e) =>
                      setNewField((prev) => ({ ...prev, name: e.target.value }))
                    }
                    onKeyDown={(e) => e.key === "Enter" && handleAdd()}
                    className="w-full h-7 bg-neutral-800 text-white px-3 py-1 rounded"
                  />
                </div>
                <div>
                  <label className="text-xs block mb-1 text-neutral-400">Type</label>
                  <select
                    value={newField.type}
                    onChange={(e) =>
                      setNewField((prev) => ({
                        ...prev,
                        type: e.target.value as FieldType,
                      }))
                    }
                    className="bg-neutral-800 text-white px-3 py-1 rounded w-full"
                  >
                    <option value="string">string</option>
                    <option value="datetime">datetime</option>
                  </select>
                </div>
              </div>
              <div className="mt-3 text-right">
                <button
                  onClick={handleAdd}
                  disabled={!newField.name.trim()}
                  className="px-3 py-1 text-sm rounded 
                    bg-neutral-700 
                    text-white 
                    disabled:opacity-20 
                    disabled:cursor-not-allowed 
                    hover:bg-neutral-600 disabled:hover:bg-neutral-700"
                >
                  Add
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
