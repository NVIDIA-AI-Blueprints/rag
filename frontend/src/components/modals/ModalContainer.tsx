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

import { createPortal } from "react-dom";
import { useCallback } from "react";

interface ModalContainerProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  maxWidth?: string;
}

const CloseIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const ModalBackdrop = ({ onClose }: { onClose: () => void }) => (
  <div
    className="absolute inset-0 bg-black bg-opacity-60"
    onClick={onClose}
  />
);

const ModalHeader = ({ title, onClose }: { title: string; onClose: () => void }) => (
  <div className="flex items-center justify-between mb-6">
    <h2 className="text-2xl font-bold">{title}</h2>
    <button
      onClick={onClose}
      className="text-gray-400 hover:text-white transition-colors p-1 rounded"
      title="Close modal"
    >
      <CloseIcon />
    </button>
  </div>
);

const ModalContent = ({ 
  children, 
  maxWidth = "max-w-2xl" 
}: { 
  children: React.ReactNode; 
  maxWidth: string;
}) => (
  <div className={`relative max-h-[90vh] w-full ${maxWidth} overflow-y-auto rounded-lg bg-nvidia-darker p-6 text-white shadow-xl border border-neutral-700`}>
    {children}
  </div>
);

export const ModalContainer = ({ 
  isOpen, 
  onClose, 
  title, 
  children, 
  maxWidth = "max-w-2xl" 
}: ModalContainerProps) => {
  const handleBackdropClick = useCallback((e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }, [onClose]);

  if (!isOpen) return null;

  const modalContent = (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={handleBackdropClick}>
      <ModalBackdrop onClose={onClose} />
      <ModalContent maxWidth={maxWidth}>
        <ModalHeader title={title} onClose={onClose} />
        {children}
      </ModalContent>
    </div>
  );

  return createPortal(modalContent, document.body);
}; 