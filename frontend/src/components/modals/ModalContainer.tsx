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

import { Modal, Text, Flex, Button } from "@kui/react";
import { X } from "lucide-react";

interface ModalContainerProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  maxWidth?: string;
}

export const ModalContainer = ({ 
  isOpen, 
  onClose, 
  title, 
  children
}: ModalContainerProps) => {
  if (!isOpen) return null;

  return (
    <Modal 
      open={isOpen} 
      onOpenChange={(open) => !open && onClose()}
      slotHeading={title}
    >
      <Flex direction="col" gap="density-lg">
        <Flex justify="between" align="center">
          <Text kind="title/md">{title}</Text>
          <Button
            onClick={onClose}
            kind="tertiary"
            size="small"
            title="Close modal"
          >
            <X size={20} />
          </Button>
        </Flex>
        {children}
      </Flex>
    </Modal>
  );
};
