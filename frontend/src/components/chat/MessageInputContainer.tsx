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

import { useState, useCallback } from "react";
import { MessageTextarea } from "./MessageTextarea";
import { MessageActions } from "./MessageActions";
import { ChatActionsMenu } from "./ChatActionsMenu";
import { ImagePreview } from "./ImagePreview";
import { Block, Flex, Text } from "@kui/react";
import { ImagePlus } from "lucide-react";
import {
  useImageAttachmentStore,
  fileToBase64,
  isValidImageFile,
  MAX_IMAGE_SIZE,
} from "../../store/useImageAttachmentStore";
import { useToastStore } from "../../store/useToastStore";

export const MessageInputContainer = () => {
  const { attachedImages, addImage } = useImageAttachmentStore();
  const { showToast } = useToastStore();
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set to false if we're actually leaving the container
    // (not just moving between child elements)
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
      setIsDragOver(false);
    }
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const files = e.dataTransfer.files;
      if (!files || files.length === 0) return;

      for (const file of Array.from(files)) {
        if (!isValidImageFile(file)) {
          showToast(`"${file.name}" is not a valid image file`, "warning");
          continue;
        }

        if (file.size > MAX_IMAGE_SIZE) {
          showToast(`"${file.name}" is too large (max 10MB)`, "warning");
          continue;
        }

        try {
          const base64 = await fileToBase64(file);
          addImage(base64, file.name);
        } catch (error) {
          console.error("Failed to read dropped image:", error);
          showToast(`Failed to read "${file.name}"`, "error");
        }
      }
    },
    [addImage, showToast]
  );

  return (
    <Flex direction="col" gap="2">
      {/* Image previews above input */}
      {attachedImages.length > 0 && <ImagePreview />}

      {/* Input container with drag & drop */}
      <Block
        style={{
          position: "relative",
          borderRadius: "8px",
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Drop zone overlay */}
        {isDragOver && (
          <Block
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "rgba(118, 185, 0, 0.1)",
              border: "2px dashed var(--nv-green)",
              borderRadius: "8px",
              pointerEvents: "none",
            }}
          >
            <Flex 
              direction="col" 
              align="center" 
              gap="density-sm"
              style={{ 
                padding: "16px",
                backgroundColor: "var(--background-color-surface-raised)",
                borderRadius: "8px",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
              }}
            >
              <ImagePlus 
                size={32} 
                style={{ color: "var(--nv-green)" }} 
              />
              <Text kind="body/bold/md" style={{ color: "var(--nv-green)" }}>
                Drop images here
              </Text>
              <Text kind="body/regular/xs" style={{ color: "var(--text-color-subtle)" }}>
                JPEG, PNG, GIF, WebP up to 10MB
              </Text>
            </Flex>
          </Block>
        )}

        <MessageTextarea />
        <Block
          style={{
            position: "absolute",
            left: "12px",
            top: "50%",
            transform: "translateY(-50%)",
          }}
        >
          <ChatActionsMenu />
        </Block>
        <MessageActions />
      </Block>
    </Flex>
  );
}; 
