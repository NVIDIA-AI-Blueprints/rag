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

import { create } from "zustand";

export interface ToastMessage {
  id: string;
  message: string;
  status: "error" | "warning" | "success" | "info";
}

interface ToastState {
  toasts: ToastMessage[];
  showToast: (message: string, status?: ToastMessage["status"]) => void;
  removeToast: (id: string) => void;
}

const generateId = () => `toast-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  showToast: (message, status = "error") => {
    const id = generateId();
    set((state) => ({
      toasts: [...state.toasts, { id, message, status }],
    }));
    // Auto-remove after 5 seconds
    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      }));
    }, 5000);
  },
  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),
}));

