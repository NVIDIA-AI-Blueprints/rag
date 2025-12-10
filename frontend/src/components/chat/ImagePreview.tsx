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

import { useImageAttachmentStore, type AttachedImage } from "../../store/useImageAttachmentStore";
import { Flex, Text, Button, Block, Panel } from "@kui/react";

const RemoveIcon = () => (
  <svg
    style={{ width: "16px", height: "16px" }}
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);

/**
 * Single image preview item.
 */
const ImagePreviewItem = ({ image, onRemove }: { image: AttachedImage; onRemove: (id: string) => void }) => (
  <Panel density="compact" style={{ width: "fit-content" }}>
    <Flex gap="density-md" align="start">
      {/* Thumbnail */}
      <Block style={{ flexShrink: 0 }}>
        <Flex
          align="center"
          justify="center"
          style={{
            width: "40px",
            height: "40px",
            borderRadius: "var(--radius-md)",
            overflow: "hidden",
            border: "1px solid var(--border-color-base)",
          }}
        >
          <img
            src={image.dataUri}
            alt={image.name}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
          />
        </Flex>
      </Block>

      {/* Filename */}
      <Text
        kind="body/regular/sm"
        style={{
          maxWidth: "20ch",
          wordBreak: "break-word",
        }}
      >
        {image.name}
      </Text>

      {/* Remove button */}
      <Button
        onClick={() => onRemove(image.id)}
        kind="tertiary"
        color="neutral"
        size="tiny"
        title="Remove image"
      >
        <RemoveIcon />
      </Button>
    </Flex>
  </Panel>
);

/**
 * Displays previews of all attached images.
 */
export const ImagePreview = () => {
  const { attachedImages, removeImage } = useImageAttachmentStore();

  if (attachedImages.length === 0) return null;

  return (
    <Flex gap="density-sm" wrap="wrap">
      {attachedImages.map((image) => (
        <ImagePreviewItem key={image.id} image={image} onRemove={removeImage} />
      ))}
    </Flex>
  );
};

