'use client';

import React, { useMemo, useState } from 'react';

type Chat = {
  id: string;
  title: string;
  snippet: string;
  updatedAt: Date;
  pinned: boolean;
  unread: number;
  mode: 1 | 2 | 3 | 4;
};

export default function Page() {
  const [selectedMode, setSelectedMode] = useState<1 | 2 | 3 | 4>(1);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [chats, setChats] = useState<Chat[]>(() => makeFakeChats());

  const pinned = useMemo(
    () =>
      chats
        .filter((c) => c.pinned)
        .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()),
    [chats]
  );
  const recent = useMemo(
    () =>
      chats
        .filter((c) => !c.pinned)
        .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
        .slice(0, 10),
    [chats]
  );
  const allSorted = useMemo(
    () =>
      [...chats].sort(
        (a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()
      ),
    [chats]
  );

  const selectedChat = useMemo(
    () => chats.find((c) => c.id === selectedChatId) ?? null,
    [chats, selectedChatId]
  );

  function handleNewChat() {
    const id = crypto.randomUUID();
    const newChat: Chat = {
      id,
      title: `New ${modeName(selectedMode)} chat`,
      snippet: 'Say hello to your new conversation.',
      updatedAt: new Date(),
      pinned: false,
      unread: 0,
      mode: selectedMode,
    };
    setChats((prev) => [newChat, ...prev]);
    setSelectedChatId(id);
  }

  function togglePin(id: string) {
    setChats((prev) =>
      prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c))
    );
  }

  function openChat(id: string) {
    setSelectedChatId(id);
    // Mark as read, update timestamp a bit for demo feel.
    setChats((prev) =>
      prev.map((c) =>
        c.id === id
          ? {
              ...c,
              unread: 0,
              updatedAt: new Date(),
              snippet:
                c.snippet === 'Say hello to your new conversation.'
                  ? c.snippet
                  : c.snippet || '…',
            }
          : c
      )
    );
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebarTop">
          <button className="newBtn" onClick={handleNewChat}>
            <PlusIcon />
            <span>New Chat</span>
          </button>

          <div className="modes">
            {[1, 2, 3, 4].map((m) => (
              <button
                key={m}
                className={`modeTile ${
                  selectedMode === (m as 1 | 2 | 3 | 4) ? 'active' : ''
                }`}
                onClick={() => setSelectedMode(m as 1 | 2 | 3 | 4)}
                aria-pressed={selectedMode === m}
                aria-label={`Mode ${m}`}
              >
                <span>{m}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebarScroll">
          <Section title="Pinned chats" count={pinned.length} />
          <ul className="list">
            {pinned.map((chat) => (
              <ChatRow
                key={chat.id}
                chat={chat}
                selected={selectedChatId === chat.id}
                onOpen={() => openChat(chat.id)}
                onTogglePin={() => togglePin(chat.id)}
              />
            ))}
            {pinned.length === 0 && (
              <EmptyRow text="No pinned chats yet." />
            )}
          </ul>

          <Section title="Recent chats" count={recent.length} />
          <ul className="list">
            {recent.map((chat) => (
              <ChatRow
                key={chat.id}
                chat={chat}
                selected={selectedChatId === chat.id}
                onOpen={() => openChat(chat.id)}
                onTogglePin={() => togglePin(chat.id)}
              />
            ))}
            {recent.length === 0 && <EmptyRow text="No recent chats." />}
          </ul>

          <Section title="All chats" count={allSorted.length} />
          <ul className="list">
            {allSorted.map((chat) => (
              <ChatRow
                key={chat.id}
                chat={chat}
                selected={selectedChatId === chat.id}
                onOpen={() => openChat(chat.id)}
                onTogglePin={() => togglePin(chat.id)}
              />
            ))}
            {allSorted.length === 0 && <EmptyRow text="Nothing here yet." />}
          </ul>
        </div>
      </aside>

      <main className="main">
        {selectedChat ? (
          <div className="chatContainer">
            <div className="chatHeader">
              <div className="chatHeaderMeta">
                <Avatar seed={selectedChat.title} />
                <div className="chatHeaderText">
                  <div className="chatTitle">{selectedChat.title}</div>
                  <div className="chatSub">
                    Mode {selectedChat.mode} •{' '}
                    {formatTime(selectedChat.updatedAt)}
                  </div>
                </div>
              </div>
              <button
                className={`pinBtn ${selectedChat.pinned ? 'on' : ''}`}
                onClick={() => togglePin(selectedChat.id)}
                aria-label={
                  selectedChat.pinned ? 'Unpin chat' : 'Pin chat'
                }
                title={selectedChat.pinned ? 'Unpin' : 'Pin'}
              >
                <PinIcon />
              </button>
            </div>

            <div className="messages">
              <Message
                from="you"
                text="Hey, can you help me draft a workout that fits around
                nap schedules?"
                time="7:32 PM"
              />
              <Message
                from="ai"
                text="Absolutely. Quick questions: equipment available,
                days per week, and any movements you love or hate?"
                time="7:33 PM"
              />
              <Message
                from="you"
                text="Dumbbells and a stroller. 4x/week. Love squats,
                hate burpees."
                time="7:34 PM"
              />
            </div>

            <div className="composer">
              <input
                className="input"
                placeholder="Message T3 Chat…"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    // no-op in demo
                  }
                }}
              />
              <button className="sendBtn" title="Send">
                <SendIcon />
              </button>
            </div>
          </div>
        ) : (
          <div className="emptyMain">
            <div className="emptyCard">
              <div className="emptyTitle">Start a conversation</div>
              <div className="emptySub">
                Pick a mode on the left and hit New Chat.
              </div>
            </div>
          </div>
        )}
      </main>

      <style jsx>{`
        :root {
          --bg: #0b1020;
          --bg-2: #0e1427;
          --panel: #0f1629;
          --panel-2: #0f1b33;
          --border: rgba(255, 255, 255, 0.08);
          --text: #e6e9f2;
          --muted: #a7b1c2;
          --muted-2: #7f8aa3;
          --accent: #6aa1ff;
          --accent-2: #8b5cf6;
          --success: #22c55e;
          --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }

        .app {
          display: flex;
          height: 100dvh;
          background: radial-gradient(
              1200px 1200px at 10% -10%,
              rgba(108, 99, 255, 0.08),
              transparent 60%
            ),
            radial-gradient(
              800px 800px at 100% 30%,
              rgba(99, 179, 237, 0.08),
              transparent 60%
            ),
            var(--bg);
          color: var(--text);
        }

        .sidebar {
          width: 320px;
          display: flex;
          flex-direction: column;
          border-right: 1px solid var(--border);
          background: linear-gradient(180deg, var(--panel), var(--panel-2));
        }

        .sidebarTop {
          padding: 16px;
          gap: 12px;
          display: grid;
        }

        .newBtn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          height: 44px;
          border-radius: 12px;
          padding: 0 14px;
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: linear-gradient(90deg, var(--accent), var(--accent-2));
          box-shadow: var(--shadow);
          transition: transform 120ms ease, box-shadow 120ms ease,
            filter 150ms ease;
        }
        .newBtn:hover {
          transform: translateY(-1px);
          filter: brightness(1.05);
          box-shadow: 0 14px 40px rgba(0, 0, 0, 0.4);
        }
        .newBtn:active {
          transform: translateY(0);
          filter: brightness(0.98);
        }

        .modes {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 10px;
        }

        .modeTile {
          position: relative;
          border-radius: 12px;
          height: 70px;
          border: 1px solid var(--border);
          background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0.03),
            rgba(0, 0, 0, 0.12)
          );
          color: var(--text);
          display: grid;
          place-items: center;
          font-weight: 700;
          font-size: 18px;
          transition: border-color 150ms ease, transform 120ms ease,
            background 200ms ease, box-shadow 150ms ease;
        }
        .modeTile:hover {
          border-color: rgba(255, 255, 255, 0.14);
          transform: translateY(-1px);
          box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
        }
        .modeTile.active {
          background: linear-gradient(
              180deg,
              rgba(99, 179, 237, 0.12),
              rgba(139, 92, 246, 0.12)
            ),
            linear-gradient(180deg, rgba(255, 255, 255, 0.04), transparent);
          border-color: rgba(139, 92, 246, 0.45);
        }

        .sidebarScroll {
          flex: 1;
          overflow-y: auto;
          padding: 8px 12px 16px 12px;
          scrollbar-width: thin;
          scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
        }
        .sidebarScroll::-webkit-scrollbar {
          width: 8px;
        }
        .sidebarScroll::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.15);
          border-radius: 8px;
        }

        .section {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 6px;
          margin-top: 6px;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 12px;
          letter-spacing: 0.08em;
        }
        .section .count {
          font-variant-numeric: tabular-nums;
          opacity: 0.8;
        }

        .list {
          list-style: none;
          padding: 0;
          margin: 0 0 10px 0;
          display: grid;
          gap: 4px;
        }

        .row {
          display: grid;
          grid-template-columns: 40px 1fr auto;
          gap: 10px;
          align-items: center;
          padding: 10px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0.02),
            rgba(0, 0, 0, 0.1)
          );
          transition: transform 120ms ease, background 150ms ease,
            border-color 150ms ease, box-shadow 150ms ease;
          cursor: pointer;
        }
        .row:hover {
          transform: translateY(-1px);
          border-color: rgba(255, 255, 255, 0.14);
          box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
        }
        .row.selected {
          border-color: rgba(99, 179, 237, 0.45);
          box-shadow: 0 10px 28px rgba(67, 129, 211, 0.25);
          background: linear-gradient(
            180deg,
            rgba(99, 179, 237, 0.06),
            rgba(139, 92, 246, 0.06)
          );
        }

        .meta {
          display: grid;
          gap: 2px;
        }
        .titleLine {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .title {
          font-weight: 600;
          font-size: 14px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .badges {
          display: inline-flex;
          gap: 6px;
          align-items: center;
        }
        .modeBadge {
          padding: 2px 6px;
          border-radius: 8px;
          font-size: 11px;
          color: #b4cffd;
          border: 1px solid rgba(106, 161, 255, 0.35);
          background: rgba(106, 161, 255, 0.12);
        }
        .unread {
          padding: 2px 6px;
          border-radius: 999px;
          font-size: 11px;
          color: white;
          background: linear-gradient(90deg, #ef4444, #f97316);
          box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35);
        }

        .snippet {
          font-size: 12px;
          color: var(--muted);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .right {
          display: grid;
          justify-items: end;
          align-items: center;
          gap: 8px;
        }
        .time {
          font-size: 12px;
          color: var(--muted-2);
          font-variant-numeric: tabular-nums;
        }
        .pinIconBtn {
          border: none;
          background: transparent;
          padding: 0;
          opacity: 0.7;
          transition: opacity 120ms ease, transform 120ms ease;
          cursor: pointer;
        }
        .pinIconBtn:hover {
          opacity: 1;
          transform: translateY(-1px);
        }

        .emptyRow {
          padding: 16px 12px;
          color: var(--muted-2);
          font-size: 13px;
        }

        .main {
          flex: 1;
          display: grid;
          grid-template-rows: auto 1fr auto;
          background: linear-gradient(180deg, var(--bg-2), #0a0f1f);
        }

        .chatContainer {
          display: grid;
          grid-template-rows: auto 1fr auto;
          height: 100%;
        }

        .chatHeader {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 14px 18px;
          border-bottom: 1px solid var(--border);
          backdrop-filter: blur(8px);
          background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0.02),
            rgba(255, 255, 255, 0.01)
          );
        }
        .chatHeaderMeta {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .chatHeaderText {
          display: grid;
          gap: 2px;
        }
        .chatTitle {
          font-weight: 700;
          font-size: 16px;
        }
        .chatSub {
          font-size: 12px;
          color: var(--muted-2);
        }
        .pinBtn {
          height: 36px;
          width: 36px;
          display: grid;
          place-items: center;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: rgba(255, 255, 255, 0.02);
          transition: transform 120ms ease, border-color 150ms ease,
            background 150ms ease;
          cursor: pointer;
        }
        .pinBtn:hover {
          transform: translateY(-1px);
          border-color: rgba(255, 255, 255, 0.14);
        }
        .pinBtn.on {
          border-color: rgba(99, 179, 237, 0.4);
          background: rgba(99, 179, 237, 0.12);
        }

        .messages {
          padding: 18px;
          display: grid;
          gap: 14px;
          overflow-y: auto;
        }

        .msg {
          display: grid;
          grid-template-columns: 36px 1fr;
          gap: 10px;
          align-items: start;
        }
        .bubble {
          border-radius: 14px;
          padding: 10px 12px;
          line-height: 1.5;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid var(--border);
        }
        .metaLine {
          margin-top: 4px;
          color: var(--muted-2);
          font-size: 12px;
        }

        .composer {
          display: flex;
          gap: 10px;
          padding: 12px;
          border-top: 1px solid var(--border);
          background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0.02),
            rgba(0, 0, 0, 0.12)
          );
        }
        .input {
          flex: 1;
          height: 44px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: rgba(255, 255, 255, 0.02);
          color: var(--text);
          padding: 0 12px;
          outline: none;
          transition: border-color 120ms ease, background 120ms ease;
        }
        .input:focus {
          border-color: rgba(106, 161, 255, 0.45);
          background: rgba(106, 161, 255, 0.06);
        }
        .sendBtn {
          height: 44px;
          width: 44px;
          display: grid;
          place-items: center;
          border-radius: 12px;
          border: 1px solid var(--border);
          color: white;
          background: linear-gradient(
            90deg,
            rgba(106, 161, 255, 0.6),
            rgba(139, 92, 246, 0.6)
          );
          transition: transform 120ms ease, filter 150ms ease;
        }
        .sendBtn:hover {
          transform: translateY(-1px);
          filter: brightness(1.05);
        }

        .emptyMain {
          display: grid;
          place-items: center;
        }
        .emptyCard {
          border: 1px solid var(--border);
          background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0.02),
            rgba(0, 0, 0, 0.12)
          );
          border-radius: 16px;
          padding: 24px 28px;
          text-align: center;
          box-shadow: var(--shadow);
        }
        .emptyTitle {
          font-size: 18px;
          font-weight: 700;
          margin-bottom: 6px;
        }
        .emptySub {
          color: var(--muted);
          font-size: 14px;
        }

        @media (max-width: 900px) {
          .sidebar {
            width: 290px;
          }
        }
        @media (max-width: 720px) {
          .sidebar {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}

function ChatRow(props: {
  chat: Chat;
  selected: boolean;
  onOpen: () => void;
  onTogglePin: () => void;
}) {
  const { chat, selected, onOpen, onTogglePin } = props;
  return (
    <li
      className={`row ${selected ? 'selected' : ''}`}
      onClick={onOpen}
      role="button"
      aria-pressed={selected}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter') onOpen();
        if (e.key === 'p') onTogglePin();
      }}
    >
      <Avatar seed={chat.title} />
      <div className="meta">
        <div className="titleLine">
          <div className="title">{chat.title}</div>
          <div className="badges">
            <span className="modeBadge">Mode {chat.mode}</span>
            {chat.unread > 0 && (
              <span className="unread">{chat.unread}</span>
            )}
          </div>
        </div>
        <div className="snippet">{chat.snippet}</div>
      </div>
      <div className="right">
        <div className="time">{formatTime(chat.updatedAt)}</div>
        <button
          className="pinIconBtn"
          onClick={(e) => {
            e.stopPropagation();
            onTogglePin();
          }}
          aria-label={chat.pinned ? 'Unpin chat' : 'Pin chat'}
          title={chat.pinned ? 'Unpin' : 'Pin'}
        >
          <PinIcon filled={chat.pinned} />
        </button>
      </div>
    </li>
  );
}

function Section(props: { title: string; count?: number }) {
  return (
    <div className="section">
      <span>{props.title}</span>
      {typeof props.count === 'number' && (
        <span className="count">{props.count}</span>
      )}
      <style jsx>{`
        .section .count {
          font-variant-numeric: tabular-nums;
        }
      `}</style>
    </div>
  );
}

function Avatar({ seed }: { seed: string }) {
  const hue = hashHue(seed);
  const bg = `hsl(${hue} 70% 42% / 1)`;
  const fg = 'white';
  const letter = (seed?.trim()?.[0] || 'C').toUpperCase();
  return (
    <div
      style={{
        height: 36,
        width: 36,
        borderRadius: 10,
        display: 'grid',
        placeItems: 'center',
        fontWeight: 800,
        color: fg,
        background:
          'linear-gradient(180deg, rgba(255,255,255,0.12), rgba(0,0,0,0.12)),' +
          bg,
        border: '1px solid rgba(255,255,255,0.15)',
      }}
      aria-hidden
    >
      {letter}
    </div>
  );
}

function Message(props: {
  from: 'you' | 'ai';
  text: string;
  time: string;
}) {
  return (
    <div className="msg">
      <Avatar seed={props.from === 'you' ? 'You' : 'AI'} />
      <div>
        <div className="bubble">{props.text}</div>
        <div className="metaLine">
          {props.from === 'you' ? 'You' : 'T3 Chat'} • {props.time}
        </div>
      </div>
      <style jsx>{`
        .bubble {
          white-space: pre-wrap;
        }
      `}</style>
    </div>
  );
}

function PlusIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <path
        d="M12 5v14M5 12h14"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function PinIcon({ filled }: { filled?: boolean }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill={filled ? 'currentColor' : 'none'}
      aria-hidden
    >
      <path
        d="M9.5 3.75h5l.75 6L20 14l-6 1.25L12 21l-2-5.75L4 14l4.5-4.25.75-6z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <path
        d="M4 12l15-8-4 8 4 8L4 12z"
        stroke="white"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function formatTime(d: Date): string {
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return 'now';
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hrs = Math.floor(min / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d`;
  return d.toLocaleDateString();
}

function hashHue(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h % 360);
}

function modeName(m: 1 | 2 | 3 | 4) {
  switch (m) {
    case 1:
      return 'General';
    case 2:
      return 'Code';
    case 3:
      return 'Analysis';
    case 4:
      return 'Creative';
  }
}

function makeFakeChats(): Chat[] {
  const titles = [
    'Weekly lift planning',
    'Data viz ideas',
    'LLM prompt tuning',
    'Rust borrow checker q’s',
    'Golang API design',
    'Nutrition macros',
    'TypeScript types',
    'Model eval metrics',
    'Postgres indexing',
    'Unix text fu',
    'Stroller WODs',
    'Next.js routing',
    'Gradient clipping',
    'CUDA basics',
    'K8s rollout',
  ];
  const snippets = [
    'Let’s map out a balanced split.',
    'Which charts fit this story?',
    'Try few-shot with structure.',
    'Lifetimes or Rc here?',
    'AuthN/AuthZ patterns?',
    'Lean bulk targets?',
    'Narrow type with generics.',
    'ROC vs PR curves?',
    'GIN vs BTREE here.',
    'awk/sed one-liners.',
    'Park laps between sets.',
    'App router nuances.',
    'Avoid exploding grads.',
    'Block/grid dims tips.',
    'Blue/green vs canary.',
  ];
  const now = Date.now();
  const out: Chat[] = [];

  for (let i = 0; i < titles.length; i++) {
    out.push({
      id: crypto.randomUUID(),
      title: titles[i],
      snippet: snippets[i],
      updatedAt: new Date(
        now - (i * 7 + (i % 3) * 23 + 3000) * 60 * 1000
      ),
      pinned: i < 3,
      unread: i % 4 === 0 ? 2 : i % 7 === 0 ? 1 : 0,
      mode: ((i % 4) + 1) as 1 | 2 | 3 | 4,
    });
  }
  return out;
}

function EmptyRow({ text }: { text: string }) {
  return (
    <li className="emptyRow" role="note" aria-live="polite">
      {text}
      <style jsx>{`
        .emptyRow {
          padding: 16px 12px;
          color: var(--muted-2);
          font-size: 13px;
        }
      `}</style>
    </li>
  );
}