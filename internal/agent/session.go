package agent

import (
	"sync"
	"time"

	"myagent/internal/model"
)

// Session 独立的对话会话，每个 WebSocket 连接一个
type Session struct {
	mu           sync.Mutex
	ID           string
	AgentID      string
	History      []model.Message
	SystemPrompt string
	CreatedAt    time.Time
	LastActiveAt time.Time
	MaxTokens    int    // token 预算上限（0 表示使用默认值）
	Summary      string // 滚动摘要（方案 A）：历史压缩后累积的增量摘要，始终只保留一条
}

// NewSession 创建新会话
func NewSession(id, agentID, systemPrompt string, maxTokens int) *Session {
	now := time.Now()
	return &Session{
		ID:           id,
		AgentID:      agentID,
		SystemPrompt: systemPrompt,
		CreatedAt:    now,
		LastActiveAt: now,
		MaxTokens:    maxTokens,
		History: []model.Message{
			{Role: "system", Content: systemPrompt},
		},
	}
}

// AddMessage 添加消息到会话历史
func (s *Session) AddMessage(msg model.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.History = append(s.History, msg)
	s.LastActiveAt = time.Now()
}

// GetHistory 获取会话历史（线程安全拷贝）
func (s *Session) GetHistory() []model.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	copied := make([]model.Message, len(s.History))
	copy(copied, s.History)
	return copied
}

// SetHistory 设置会话历史
func (s *Session) SetHistory(history []model.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.History = history
	s.LastActiveAt = time.Now()
}

// GetSummary 获取滚动摘要（线程安全）
func (s *Session) GetSummary() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.Summary
}

// SetSummary 设置滚动摘要（线程安全）
func (s *Session) SetSummary(summary string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Summary = summary
}

// Clear 清空历史与滚动摘要，仅保留 system prompt
func (s *Session) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.History = []model.Message{
		{Role: "system", Content: s.SystemPrompt},
	}
	s.Summary = ""
	s.LastActiveAt = time.Now()
}

// EstimateTokens 估算当前历史的 token 数
// 简单估算：中文约 2 字符/token，英文约 4 字符/token，取平均 3 字符/token
func (s *Session) EstimateTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return estimateMessagesTokens(s.History)
}

func estimateMessagesTokens(messages []model.Message) int {
	total := 0
	for _, msg := range messages {
		total += estimateStringTokens(msg.Content)
		for _, tc := range msg.ToolCalls {
			total += estimateStringTokens(tc.Function.Name)
			total += estimateStringTokens(tc.Function.Arguments)
		}
		// 每条消息有固定开销（role 等元数据）
		total += 4
	}
	return total
}

func estimateStringTokens(s string) int {
	// 混合中英文场景取平均：约 3 字符 = 1 token
	runeCount := len([]rune(s))
	if runeCount == 0 {
		return 0
	}
	return (runeCount + 2) / 3
}

// SessionManager 管理多个会话
//
// 基于 sync.Map 实现：读多写少、key（sessionID）互不相交的场景下，
// 读路径完全无锁（Load 走 read 快照 + CAS），写路径仅在新增 key 时加锁。
// 每个 WebSocket 连接只读写自己的 session，天然符合 sync.Map 的适用场景。
type SessionManager struct {
	sessions sync.Map // key: sessionID, value: *Session
}

// NewSessionManager 创建会话管理器
func NewSessionManager() *SessionManager {
	return &SessionManager{}
}

// Get 获取会话（无锁读路径）
func (sm *SessionManager) Get(sessionID string) (*Session, bool) {
	v, ok := sm.sessions.Load(sessionID)
	if !ok {
		return nil, false
	}
	return v.(*Session), true
}

// Create 创建会话
func (sm *SessionManager) Create(id, agentID, systemPrompt string, maxTokens int) *Session {
	s := NewSession(id, agentID, systemPrompt, maxTokens)
	sm.sessions.Store(id, s)
	return s
}

// Delete 删除会话
func (sm *SessionManager) Delete(sessionID string) {
	sm.sessions.Delete(sessionID)
}

// ListByAgent 列出指定 agent 的所有会话
// 注意：Range 遍历不保证一致性快照，回调内不得修改 map（先收集后处理）
func (sm *SessionManager) ListByAgent(agentID string) []*Session {
	var result []*Session
	sm.sessions.Range(func(_, v interface{}) bool {
		if s, ok := v.(*Session); ok && s.AgentID == agentID {
			result = append(result, s)
		}
		return true
	})
	return result
}

// CleanExpired 清理过期会话（超过 maxAge 未活跃的会话）
// 先收集过期的 sessionID，再统一删除，避免在 Range 回调内修改 map
func (sm *SessionManager) CleanExpired(maxAge time.Duration) int {
	cutoff := time.Now().Add(-maxAge)
	var expired []string
	sm.sessions.Range(func(key, v interface{}) bool {
		if s, ok := v.(*Session); ok && s.LastActiveAt.Before(cutoff) {
			expired = append(expired, key.(string))
		}
		return true
	})
	for _, id := range expired {
		sm.sessions.Delete(id)
	}
	return len(expired)
}
