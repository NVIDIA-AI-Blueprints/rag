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

/**
 * Represents an attached image.
 */
export interface AttachedImage {
  id: string;
  dataUri: string;
  name: string;
}

/**
 * State interface for the image attachment store.
 */
interface ImageAttachmentState {
  /** Array of attached images */
  attachedImages: AttachedImage[];
  /** Add an image */
  addImage: (dataUri: string, filename: string) => void;
  /** Remove an image by id */
  removeImage: (id: string) => void;
  /** Clear all attached images */
  clearAllImages: () => void;
}

/**
 * Generate a unique ID for images.
 */
const generateId = () => `img-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

/**
 * Zustand store for managing image attachment state in chat.
 */
export const useImageAttachmentStore = create<ImageAttachmentState>((set) => ({
  attachedImages: [],
  addImage: (dataUri, filename) => set((state) => ({
    attachedImages: [...state.attachedImages, { id: generateId(), dataUri, name: filename }]
  })),
  removeImage: (id) => set((state) => ({
    attachedImages: state.attachedImages.filter((img) => img.id !== id)
  })),
  clearAllImages: () => set({ attachedImages: [] }),
}));

/**
 * Convert a File to base64 data URI.
 */
export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

/**
 * Validate that a file is an acceptable image type.
 */
export const isValidImageFile = (file: File): boolean => {
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
  return validTypes.includes(file.type);
};

/**
 * Maximum image size in bytes (10MB).
 */
export const MAX_IMAGE_SIZE = 10 * 1024 * 1024;

