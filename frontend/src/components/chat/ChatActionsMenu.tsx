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

import { useState, useRef } from "react";
import { useChatStore } from "../../store/useChatStore";
import { useImageAttachmentStore, fileToBase64, isValidImageFile, MAX_IMAGE_SIZE } from "../../store/useImageAttachmentStore";
import { useToastStore } from "../../store/useToastStore";
import { Dropdown, Modal, Button, Flex, Text } from "@kui/react";

interface ChatActionItem {
  children: string;
  slotLeft?: React.ReactNode;
  disabled?: boolean;
  danger?: boolean;
  onSelect: (event: Event) => void;
}

// Type guard to check if an item has our onSelect method
const hasOnSelectMethod = (item: unknown): item is ChatActionItem => {
  return typeof item === 'object' && item !== null && typeof (item as ChatActionItem).onSelect === 'function';
};

const PlusIcon = () => (
  <svg style={{ width: '14px', height: '14px' }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 5v14m7-7H5" />
  </svg>
);

const TrashIcon = () => (
  <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
  </svg>
);

const ImageIcon = () => (
  <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z" />
  </svg>
);

export const ChatActionsMenu = () => {
  const { messages, clearMessages } = useChatStore();
  const { addImage } = useImageAttachmentStore();
  const { showToast } = useToastStore();
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClearChatRequest = () => {
    if (messages.length > 0) {
      setShowConfirmModal(true);
    }
  };

  const handleConfirmClear = () => {
    clearMessages();
    setShowConfirmModal(false);
  };

  const handleCancelClear = () => {
    setShowConfirmModal(false);
  };

  const handleUploadImage = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    for (const file of Array.from(files)) {
      if (!isValidImageFile(file)) {
        showToast(`"${file.name}" is not a valid file`, "warning");
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
        console.error("Failed to read image file:", error);
        showToast(`Failed to read "${file.name}"`, "error");
      }
    }

    // Reset input so the same file can be selected again
    e.target.value = "";
  };

  const hasMessages = messages.length > 0;

  const dropdownItems: ChatActionItem[] = [
    {
      children: "Add image",
      slotLeft: <ImageIcon />,
      disabled: false,
      danger: false,
      onSelect: handleUploadImage
    },
    {
      children: "Clear chat",
      slotLeft: <TrashIcon />,
      disabled: !hasMessages,
      danger: true,
      onSelect: handleClearChatRequest
    }
  ];

  return (
    <>
      {/* Hidden file input for image upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png,image/gif,image/webp"
        multiple
        onChange={handleFileChange}
        style={{ display: "none" }}
        aria-label="Upload image"
      />

      <Dropdown
        items={dropdownItems}
        size="small"
        side="top"
        align="start"
        aria-label="Chat options"
        onItemSelect={(event, item) => {
          if (hasOnSelectMethod(item)) {
            item.onSelect(event);
          }
        }}
        style={{
          color: 'var(--text-color-subtle)'
        }}
        attributes={{
          DropdownContent: {
            style: {
              marginBottom: '8px'
            }
          }
        }}
      >
        <PlusIcon />
      </Dropdown>

      <Modal
        open={showConfirmModal}
        onOpenChange={setShowConfirmModal}
        slotHeading="Clear Chat"
        slotFooter={
          <Flex align="center" justify="end" gap="density-sm">
            <Button kind="tertiary" onClick={handleCancelClear}>
              Cancel
            </Button>
            <Button color="danger" onClick={handleConfirmClear}>
              Clear Chat
            </Button>
          </Flex>
        }
      >
        <Text>
          Are you sure you want to clear all chat messages? This action cannot be undone.
        </Text>
      </Modal>
    </>
  );
}; 