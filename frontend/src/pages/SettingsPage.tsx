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

import { useMemo } from "react";
import { VerticalNav, Flex, PageHeader, Divider, Block } from "@kui/react";
import { useSettingsSections } from "../hooks/useSettingsSections";
import { useFeatureWarning } from "../hooks/useFeatureWarning";
import { SettingsContent } from "../components/settings/SettingsContent";
import { FeatureWarningModal } from "../components/modals/FeatureWarningModal";
import { 
  ICON_rag, 
  ICON_features, 
  ICON_models, 
  ICON_endpoints, 
  ICON_advanced 
} from "../components/icons";

/**
 * Settings page component providing comprehensive application configuration.
 * 
 * Built with 100% KUI components for consistent design and theming.
 * Uses Grid layout with VerticalNav sidebar and Card-based content sections.
 * 
 * @returns Settings page using KUI design system components
 */
export default function SettingsPage() {
  const { activeSection, setSection } = useSettingsSections();
  const { showModal, showWarning, confirmChange, cancelChange } = useFeatureWarning();

  const navigationItems = useMemo(() => [
    {
      id: 'ragConfig',
      slotLabel: 'RAG Configuration',
      slotIcon: <ICON_rag />,
      active: activeSection === 'ragConfig',
      href: '#ragConfig',
      attributes: {
        VerticalNavLink: {
          onClick: (e: React.MouseEvent) => {
            e.preventDefault();
            setSection('ragConfig');
          }
        }
      }
    },
    {
      id: 'features',
      slotLabel: 'Feature Toggles',
      slotIcon: <ICON_features />,
      active: activeSection === 'features',
      href: '#features',
      attributes: {
        VerticalNavLink: {
          onClick: (e: React.MouseEvent) => {
            e.preventDefault();
            setSection('features');
          }
        }
      }
    },
    {
      id: 'models',
      slotLabel: 'Model Configuration',
      slotIcon: <ICON_models />,
      active: activeSection === 'models',
      href: '#models',
      attributes: {
        VerticalNavLink: {
          onClick: (e: React.MouseEvent) => {
            e.preventDefault();
            setSection('models');
          }
        }
      }
    },
    {
      id: 'endpoints',
      slotLabel: 'Endpoint Configuration',
      slotIcon: <ICON_endpoints />,
      active: activeSection === 'endpoints',
      href: '#endpoints',
      attributes: {
        VerticalNavLink: {
          onClick: (e: React.MouseEvent) => {
            e.preventDefault();
            setSection('endpoints');
          }
        }
      }
    },
    {
      id: 'advanced',
      slotLabel: 'Advanced Settings',
      slotIcon: <ICON_advanced />,
      active: activeSection === 'advanced',
      href: '#advanced',
      attributes: {
        VerticalNavLink: {
          onClick: (e: React.MouseEvent) => {
            e.preventDefault();
            setSection('advanced');
          }
        }
      }
    }
  ], [activeSection, setSection]);

  return (
    <Block style={{ backgroundColor: 'var(--background-color-surface-base)' }}>
      <PageHeader
        slotHeading="Settings"
        slotDescription="Configure your RAG application settings and preferences"
      />
      <Divider />
      <Flex className="h-screen">
        <VerticalNav items={navigationItems} />
        <Flex direction="col" className="flex-1">
          <SettingsContent 
            activeSection={activeSection}
            onShowWarning={showWarning}
          />
        </Flex>
        <FeatureWarningModal
          isOpen={showModal}
          onClose={cancelChange}
          onConfirm={confirmChange}
        />
      </Flex>
    </Block>
  );
} 