import type { ComponentType } from 'react';

export interface SidebarListItemProps {
  text: string;
  icon: ComponentType<{ className?: string }>;
  disabled?: boolean;
  link: string;
  chip?: {
    value: string;
    size: string;
    color: string;
  };
}
