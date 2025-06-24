// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

"use client";

import { useApp } from "../../context/AppContext";
import Image from "next/image";
import { Filter } from "./FilterInput";
import { useState, useEffect, useRef } from "react";
import FilterInput from "./FilterInput";

interface MessageInputProps {
  message: string;
  setMessage: (message: string) => void;
  onSubmit: () => void;
  onAbort?: () => void;
  isStreaming: boolean;
  onReset: () => void;
  filters: Filter[];
  setFilters: (filters: Filter[]) => void;
}

export default function MessageInput({
  message,
  setMessage,
  onSubmit,
  onAbort,
  isStreaming,
  filters,
  setFilters,
}: MessageInputProps) {
  const { selectedCollections } = useApp();
  const hasSelections = selectedCollections.length > 0;
  const [showFilterInput, setShowFilterInput] = useState(false);
  const modalRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        setShowFilterInput(false);
      }
    };

    if (showFilterInput) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showFilterInput]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) onSubmit();
    }
  };

  return (
    <div className="relative border-t border-neutral-800 p-4">
      <div className="mx-auto max-w-3xl space-y-2">
        <div className="overflow-hidden rounded-lg bg-neutral-800">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your question"
            className="scrollbar-hide max-h-[200px] min-h-[44px] w-full resize-none bg-neutral-800 px-4 py-3 text-white focus:outline-none"
            style={{ height: "auto" }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "auto";
              target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
            }}
          />

          {/* Filters row */}
          <div className="flex items-center gap-2 flex-wrap px-4 pb-1">
            {filters.map((f, i) => (
              <span
                key={i}
                className="flex items-center gap-1 text-xs bg-neutral-700 text-white px-2 py-1 rounded-full"
              >
                {f.field} {f.operator} {f.value}
                <button
                  onClick={() => {
                    const updated = [...filters];
                    updated.splice(i, 1);
                    setFilters(updated);
                  }}
                  className="ml-1 text-white hover:text-red-400"
                  aria-label="Remove filter"
                >
                  ×
                </button>
              </span>
            ))}
            <button
              onClick={() => setShowFilterInput(true)}
              className="text-xs border border-neutral-600 px-2 py-1 rounded hover:bg-neutral-700 text-white"
            >
              + Filter
            </button>
          </div>

          {/* Bottom row */}
          <div className="flex items-center justify-between px-4 py-3">
            {hasSelections ? (
              <div className="rounded-full bg-neutral-100 px-4 py-1 text-sm text-black max-w-[240px] truncate">
                <div className="flex items-center gap-1">
                  <Image
                    src="/collection.svg"
                    alt="Upload files"
                    width={24}
                    height={24}
                    className="mr-1"
                  />
                  <span className="truncate">
                    {selectedCollections.join(", ")}
                  </span>
                </div>
              </div>
            ) : (
              <div className="rounded-full border border-white px-4 py-1 text-sm text-white">
                <div className="flex items-center">
                  <Image
                    src="/collection.svg"
                    alt="Upload files"
                    width={24}
                    height={24}
                    className="mr-1 invert"
                  />
                  No Collection
                </div>
              </div>
            )}

            <button
              onClick={isStreaming ? onAbort : onSubmit}
              disabled={!message.trim() && !isStreaming}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                isStreaming
                  ? "bg-neutral-600 text-white hover:brightness-90"
                  : !message.trim()
                  ? "bg-neutral-700 text-neutral-400"
                  : "bg-[var(--nv-green)] text-white hover:brightness-90"
              }`}
            >
              {isStreaming ? "Stop" : "Send"}
            </button>
          </div>
        </div>

        <p className="text-center text-xs text-gray-500">
          Model responses may be inaccurate or incomplete. Verify critical
          information before use.
        </p>
      </div>

      {showFilterInput && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center">
          <div ref={modalRef}>
            <FilterInput
              filters={filters}
              setFilters={setFilters}
              onClose={() => setShowFilterInput(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
