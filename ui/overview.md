# UI Package Overview

This package contains the Next.js frontend application for the Internal Perplexity research platform.

## Files

### App Directory Structure

- `app/layout.tsx` - Root layout with font configuration and metadata
- `app/page.tsx` - Home page that redirects to `/chat`
- `app/chat/layout.tsx` - Chat layout with parallel routes for sidebars
- `app/chat/page.tsx` - Main chat interface page
- `app/chat/@chatsidebar/default.tsx` - Left sidebar with chat navigation and pinned chats
- `app/chat/@filesidebar/default.tsx` - Right sidebar with model picker and tools
- `app/settings/page.tsx` - User settings and profile management
- `app/admin/page.tsx` - Admin dashboard for system management

### Components

- `components/chat-interface.tsx` - Main chat interface with message display and input
- `components/model-picker.tsx` - Model selection component with search and filtering
- `components/tool-picker.tsx` - Tool management component for enabling/disabling tools
- `components/ui/` - Shadcn UI components (button, card, input, textarea, etc.)

### Configuration

- `package.json` - Dependencies and scripts using pnpm
- `components.json` - Shadcn UI configuration
- `tsconfig.json` - TypeScript configuration with strict mode
- `tailwind.config.js` - Tailwind CSS configuration
- `globals.css` - Global styles and CSS variables

## Purpose

This is a chat-based research assistant interface built with Next.js, React, and Tailwind CSS. The application provides:

1. **Chat Interface**: Real-time conversation with AI models
2. **Model Management**: Selection and configuration of different AI models
3. **Tool Integration**: Management of research tools and RAG capabilities
4. **User Management**: Settings, profiles, and admin controls
5. **Responsive Design**: Mobile-friendly interface with proper layouts

## Key Features

- **Parallel Routes**: Uses Next.js parallel routes for sidebar navigation
- **Component Architecture**: Modular components for reusability
- **Type Safety**: Full TypeScript implementation with strict mode
- **UI Framework**: Shadcn UI components with Tailwind CSS
- **State Management**: React hooks for local state management
- **Navigation**: Client-side routing with Next.js App Router

## Design Decisions

- **Layout Structure**: Three-column layout (chat sidebar, main chat, model sidebar)
- **Component Organization**: Separate components for different concerns (chat, model, tools)
- **Styling**: Tailwind CSS with Shadcn UI for consistent design system
- **TypeScript**: Strict typing for better development experience and error prevention
- **Responsive**: Mobile-first design with proper breakpoints

## Dependencies

- **Next.js 16**: React framework with App Router
- **React 19**: UI library with latest features
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn UI**: Pre-built component library
- **Lucide React**: Icon library for consistent iconography
- **TypeScript**: Type-safe JavaScript development
