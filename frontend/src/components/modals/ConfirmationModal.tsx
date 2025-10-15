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

import { Modal, Button, Flex, Text } from "@kui/react";

interface ConfirmationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  confirmColor?: "danger" | "brand" | "neutral";
  cancelText?: string;
}

export const ConfirmationModal = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = "Confirm",
  confirmColor = "danger",
  cancelText = "Cancel"
}: ConfirmationModalProps) => {
  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  return (
    <Modal
      open={isOpen}
      onOpenChange={onClose}
      slotHeading={title}
      slotFooter={
        <Flex align="center" justify="end" gap="density-sm">
          <Button kind="tertiary" onClick={onClose}>
            {cancelText}
          </Button>
          <Button color={confirmColor} onClick={handleConfirm}>
            {confirmText}
          </Button>
        </Flex>
      }
    >
      <Text>
        {message}
      </Text>
    </Modal>
  );
};
