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

import { useState } from "react";
import { useApp, IngestionTask } from "@/app/context/AppContext";

export default function PendingTasksList() {
  const { pendingTasks, removePendingTask } = useApp();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // Only show tasks that are pending or completed/failed within the last 5 minutes
  const recentTasks = pendingTasks.filter(task => {
    if (task.state === "PENDING") return true;
    
    const taskTime = new Date(task.created_at).getTime();
    const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
    return taskTime > fiveMinutesAgo;
  });

  if (recentTasks.length === 0) {
    return null;
  }

  const toggleExpand = (taskId: string) => {
    setExpanded(prev => ({
      ...prev,
      [taskId]: !prev[taskId]
    }));
  };

  const getStatusIcon = (state: IngestionTask['state']) => {
    switch (state) {
      case "PENDING": 
        return (
          <div className="flex h-4 w-4 items-center justify-center">
            <div className="h-2 w-2 animate-pulse rounded-full bg-yellow-400"></div>
          </div>
        );
      case "FINISHED": 
        return (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M6.00016 10.7799L3.22016 7.99987L2.27349 8.93987L6.00016 12.6665L14.0002 4.66654L13.0602 3.72654L6.00016 10.7799Z" fill="#22C55E"/>
          </svg>
        );
      case "FAILED": 
        return (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8.00016 6.94654L4.97349 3.91987L3.91349 4.97987L6.94016 8.00654L3.91349 11.0332L4.97349 12.0932L8.00016 9.06654L11.0268 12.0932L12.0868 11.0332L9.06016 8.00654L12.0868 4.97987L11.0268 3.91987L8.00016 6.94654Z" fill="#EF4444"/>
          </svg>
        );
      default: 
        return (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8 3.33325V8.83325L11.5 10.5833M14.6667 7.99992C14.6667 11.6818 11.6819 14.6666 8 14.6666C4.3181 14.6666 1.33333 11.6818 1.33333 7.99992C1.33333 4.31802 4.3181 1.33325 8 1.33325C11.6819 1.33325 14.6667 4.31802 14.6667 7.99992Z" stroke="#9CA3AF" strokeWidth="1.33" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        );
    }
  };

  return (
    <div className="mb-4 overflow-hidden rounded-md border border-neutral-700 bg-neutral-900">
      <div className="flex items-center justify-between border-b border-neutral-700 px-3 py-2">
        <h3 className="text-xs font-medium text-neutral-200">Background Tasks</h3>
        <div className="text-xs text-neutral-400">{recentTasks.length} active</div>
      </div>

      <div className="max-h-40 overflow-y-auto">
        {recentTasks.map((task) => (
          <div
            key={task.id}
            className="border-b border-neutral-800 px-3 py-2 last:border-b-0"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(task.state)}
                <div className="text-xs">
                  <div className="font-medium text-neutral-100">
                    {task.collection_name}
                  </div>
                  <div className="text-[10px] text-neutral-400">
                    {task.documents?.length || 0} document(s) • {task.state}
                  </div>
                </div>
              </div>
              
              <div className="flex gap-1">
                <button 
                  onClick={() => toggleExpand(task.id)}
                  className="group rounded px-1.5 py-0.5 text-[10px] text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                >
                  {expanded[task.id] ? "Hide" : "Details"}
                </button>
                {task.state !== "PENDING" && (
                  <button
                    onClick={() => removePendingTask(task.id)}
                    className="group rounded px-1.5 py-0.5 text-[10px] text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                  >
                    Dismiss
                  </button>
                )}
              </div>
            </div>
            
            {expanded[task.id] && (
              <div className="mt-2 border-t border-neutral-800 pt-2 text-[10px]">
                <div className="mb-1 pl-6 text-neutral-400">
                  Started at: {new Date(task.created_at).toLocaleTimeString()}
                </div>
                {task.documents && task.documents.length > 0 && (
                  <div className="mb-1 pl-6">
                    <span className="text-neutral-400">Files:</span>
                    <ul className="mt-0.5 text-neutral-300">
                      {task.documents.map((doc, idx) => (
                        <li key={idx} className="truncate pl-2">• {doc}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {task.result && Object.keys(task.result).length > 0 && (
                  <div className="mb-1 pl-6">
                    <span className="text-neutral-400">Result:</span>
                    <pre className="mt-0.5 max-h-20 overflow-y-auto rounded bg-neutral-800 p-1 text-[10px] text-neutral-300">
                      {JSON.stringify(task.result, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 