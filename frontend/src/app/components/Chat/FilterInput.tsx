// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState } from "react";

export interface Filter {
  field: string;
  operator: string;
  value: string;
}

interface Props {
  filters: Filter[];
  setFilters: (filters: Filter[]) => void;
  onClose: () => void;
}

export default function FilterInput({ filters, setFilters, onClose }: Props) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editValue, setEditValue] = useState<Filter>({
    field: "",
    operator: "==",
    value: "",
  });

  const [newFilter, setNewFilter] = useState<Filter>({
    field: "",
    operator: "==",
    value: "",
  });

  const handleAdd = () => {
    if (!newFilter.field.trim()) return;
    setFilters([...filters, { ...newFilter }]);
    setNewFilter({ field: "", operator: "==", value: "" });
  };

  const handleSaveEdit = (index: number) => {
    const updated = [...filters];
    updated[index] = { ...editValue };
    setFilters(updated);
    setEditingIndex(null);
  };

  const handleDelete = (index: number) => {
    setFilters(filters.filter((_, i) => i !== index));
    setEditingIndex(null);
  };

  return (
    <div className="z-50 w-[360px] rounded border border-neutral-700 bg-neutral-900 p-4 text-white shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium">Edit Filters</h3>
        <button
          onClick={onClose}
          className="text-sm text-gray-400 hover:text-white"
        >
          Close
        </button>
      </div>

      <div className="space-y-3">
        {filters.map((filter, index) =>
          editingIndex === index ? (
            <div key={index} className="border border-neutral-700 p-3 rounded bg-neutral-800 space-y-2">
              <div className="flex gap-2">
                <input
                  autoFocus
                  value={editValue.field}
                  onChange={(e) =>
                    setEditValue((prev) => ({ ...prev, field: e.target.value }))
                  }
                  className="w-1/3 rounded bg-neutral-900 px-2 py-1 text-sm"
                  placeholder="Field"
                />
                <select
                  value={editValue.operator}
                  onChange={(e) =>
                    setEditValue((prev) => ({
                      ...prev,
                      operator: e.target.value,
                    }))
                  }
                  className="w-[60px] rounded bg-neutral-900 px-1 py-1 text-sm"
                >
                  <option value="==">==</option>
                  <option value="!=">!=</option>
                  <option value=">">&gt;</option>
                  <option value=">=">&gt;=</option>
                  <option value="<">&lt;</option>
                  <option value="<=">&lt;=</option>
                </select>
                <input
                  value={editValue.value}
                  onChange={(e) =>
                    setEditValue((prev) => ({ ...prev, value: e.target.value }))
                  }
                  className="flex-1 rounded bg-neutral-900 px-2 py-1 text-sm"
                  placeholder="Value"
                />
              </div>
              <div className="flex justify-end gap-2">
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
                <button onClick={() => handleSaveEdit(index)} title="Save">
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
            <div key={index} className="border border-neutral-700 bg-neutral-800 p-2 rounded flex justify-between items-start">
              <div className="text-sm">
                <div className="text-white">{filter.field} {filter.operator} {filter.value}</div>
              </div>
              <div className="space-x-3 mt-1">
                <button
                  onClick={() => {
                    setEditingIndex(index);
                    setEditValue(filter);
                  }}
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
                <button onClick={() => handleDelete(index)} title="Delete">
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
            </div>
          )
        )}
      </div>

      <div className="border-t border-neutral-700 pt-4 mt-5 space-y-3">
        <h4 className="text-sm font-medium">Add Filter</h4>
        <div className="flex gap-2">
          <input
            value={newFilter.field}
            onChange={(e) =>
              setNewFilter((prev) => ({ ...prev, field: e.target.value }))
            }
            placeholder="Field"
            className="w-1/3 rounded bg-neutral-800 px-2 py-1 text-sm"
          />
          <select
            value={newFilter.operator}
            onChange={(e) =>
              setNewFilter((prev) => ({ ...prev, operator: e.target.value }))
            }
            className="w-[60px] rounded bg-neutral-800 px-1 py-1 text-sm"
          >
            <option value="==">==</option>
            <option value="!=">!=</option>
            <option value=">">&gt;</option>
            <option value=">=">&gt;=</option>
            <option value="<">&lt;</option>
            <option value="<=">&lt;=</option>
          </select>
          <input
            value={newFilter.value}
            onChange={(e) =>
              setNewFilter((prev) => ({ ...prev, value: e.target.value }))
            }
            placeholder="Value"
            className="flex-1 rounded bg-neutral-800 px-2 py-1 text-sm"
          />
        </div>
        <div className="flex justify-end">
          <button
            onClick={handleAdd}
            disabled={!newFilter.field.trim()}
            className="text-sm text-green-500 hover:underline disabled:opacity-30"
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
}
