package agent

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"myagent/internal/config"
	"myagent/internal/llm"
	"myagent/internal/logger"
	"myagent/internal/model"
	"myagent/internal/tool"
)

// Options Agent 配置选项
type Options struct {
	Name             string
	LLM              llm.Client
	Tools            *tool.Registry
	MaxTurns         int
	MaxHistory       int // 最大保留的历史消息条数（0 表示不限制）
	MaxTokens        int // token 预算上限（0 表示使用默认 4096）
	SystemPrompt     string
	OnHistoryChanged func() // 历史变更回调（用于持久化）
}

// Agent AI Agent
type Agent struct {
	name             string
	llm              llm.Client
	tools            *tool.Registry
	maxTurns         int
	maxHistory       int
	maxTokens        int
	systemPrompt     string
	history          []model.Message
	onHistoryChanged func()
}

// New 创建 Agent
func New(opts Options) *Agent {
	maxHistory := opts.MaxHistory
	if maxHistory <= 0 {
		maxHistory = config.DefaultMaxHistory
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = config.DefaultMaxTokens
	}
	return &Agent{
		name:             opts.Name,
		llm:              opts.LLM,
		tools:            opts.Tools,
		maxTurns:         opts.MaxTurns,
		maxHistory:       maxHistory,
		maxTokens:        maxTokens,
		systemPrompt:     opts.SystemPrompt,
		onHistoryChanged: opts.OnHistoryChanged,
		history: []model.Message{
			{Role: "system", Content: opts.SystemPrompt},
		},
	}
}

// RunInteractive 交互式运行
func (a *Agent) RunInteractive(ctx context.Context) error {
	scanner := bufio.NewScanner(os.Stdin)
	logger.Infof("Agent [%s] 启动", a.name)
	fmt.Printf("🤖 %s 已启动！输入消息开始对话，输入 /quit 退出，/clear 清空历史。\n\n", a.name)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		fmt.Print("You> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		switch input {
		case "/quit", "/exit":
			logger.Infof("用户退出")
			fmt.Println("再见！")
			return nil
		case "/clear":
			a.history = []model.Message{
				{Role: "system", Content: a.systemPrompt},
			}
			fmt.Println("对话历史已清空。")
			continue
		case "/help":
			printHelp()
			continue
		}

		response, err := a.Chat(ctx, input)
		if err != nil {
			logger.Errorf("对话出错: %v", err)
			fmt.Printf("❌ 错误: %v\n\n", err)
			continue
		}

		fmt.Printf("\n%s> %s\n\n", a.name, response)
	}

	return scanner.Err()
}

// Chat 处理单次对话（支持多轮工具调用）
func (a *Agent) Chat(ctx context.Context, userMessage string) (string, error) {
	return a.ChatWithEvents(ctx, userMessage, nil)
}

// ChatWithEvents 处理单次对话，通过回调发送中间事件
func (a *Agent) ChatWithEvents(ctx context.Context, userMessage string, onEvent func(Event)) (string, error) {
	emit := func(e Event) {
		if onEvent != nil {
			onEvent(e)
		}
	}

	logger.Infof("用户输入: %s", userMessage)
	a.history = append(a.history, model.Message{
		Role:    "user",
		Content: userMessage,
	})

	toolDefs := a.tools.Definitions()

	for turn := 0; turn < a.maxTurns; turn++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		emit(Event{Type: EventThinking})

		logger.Debugf("LLM 请求: turn=%d, messages=%d", turn, len(a.history))
		resp, err := a.llm.Chat(ctx, a.history, toolDefs)
		if err != nil {
			logger.Errorf("LLM 调用失败: %v", err)
			return "", fmt.Errorf("LLM 调用失败: %w", err)
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("LLM 返回空响应")
		}

		choice := resp.Choices[0]
		a.history = append(a.history, choice.Message)

		// 如果没有工具调用，裁剪历史并返回
		if len(choice.Message.ToolCalls) == 0 {
			logger.Infof("LLM 响应: %s", truncate(choice.Message.Content, 200))
			a.trimHistory()
			a.notifyHistoryChanged()
			emit(Event{Type: EventAssistant, Content: choice.Message.Content})
			return choice.Message.Content, nil
		}

		// 处理工具调用
		fmt.Printf("  🔧 正在调用工具...\n")
		for _, tc := range choice.Message.ToolCalls {
			fmt.Printf("    → %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
			logger.Infof("工具调用: %s args=%s", tc.Function.Name, tc.Function.Arguments)
			emit(Event{Type: EventToolCall, Name: tc.Function.Name, Args: tc.Function.Arguments})

			result, err := a.tools.Execute(ctx, tc.Function.Name, tc.Function.Arguments)
			if err != nil {
				logger.Errorf("工具 %s 执行失败: %v", tc.Function.Name, err)
				result = fmt.Sprintf("工具执行错误: %v", err)
			} else {
				logger.Debugf("工具 %s 结果: %s", tc.Function.Name, truncate(result, 200))
			}
			emit(Event{Type: EventToolResult, Name: tc.Function.Name, Result: result})

			a.history = append(a.history, model.Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}

	msg := "达到最大工具调用轮次限制"
	emit(Event{Type: EventAssistant, Content: msg})
	return msg, nil
}

// ChatSession 基于独立会话的对话（支持会话隔离）
func (a *Agent) ChatSession(ctx context.Context, session *Session, userMessage string, onEvent func(Event)) (string, error) {
	emit := func(e Event) {
		if onEvent != nil {
			onEvent(e)
		}
	}

	logger.Infof("[session=%s] 用户输入: %s", session.ID, userMessage)
	session.AddMessage(model.Message{Role: "user", Content: userMessage})

	toolDefs := a.tools.Definitions()

	for turn := 0; turn < a.maxTurns; turn++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		emit(Event{Type: EventThinking})

		history := session.GetHistory()
		logger.Debugf("[session=%s] LLM 请求: turn=%d, messages=%d", session.ID, turn, len(history))
		resp, err := a.llm.Chat(ctx, history, toolDefs)
		if err != nil {
			logger.Errorf("LLM 调用失败: %v", err)
			return "", fmt.Errorf("LLM 调用失败: %w", err)
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("LLM 返回空响应")
		}

		choice := resp.Choices[0]
		session.AddMessage(choice.Message)

		if len(choice.Message.ToolCalls) == 0 {
			logger.Infof("[session=%s] LLM 响应: %s", session.ID, truncate(choice.Message.Content, 200))
			// token 预算裁剪
			a.trimSessionByTokens(ctx, session)
			emit(Event{Type: EventAssistant, Content: choice.Message.Content})
			return choice.Message.Content, nil
		}

		// 处理工具调用
		for _, tc := range choice.Message.ToolCalls {
			logger.Infof("[session=%s] 工具调用: %s args=%s", session.ID, tc.Function.Name, tc.Function.Arguments)
			emit(Event{Type: EventToolCall, Name: tc.Function.Name, Args: tc.Function.Arguments})

			result, err := a.tools.Execute(ctx, tc.Function.Name, tc.Function.Arguments)
			if err != nil {
				logger.Errorf("工具 %s 执行失败: %v", tc.Function.Name, err)
				result = fmt.Sprintf("工具执行错误: %v", err)
			} else {
				logger.Debugf("工具 %s 结果: %s", tc.Function.Name, truncate(result, 200))
			}
			emit(Event{Type: EventToolResult, Name: tc.Function.Name, Result: result})

			session.AddMessage(model.Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}

	msg := "达到最大工具调用轮次限制"
	emit(Event{Type: EventAssistant, Content: msg})
	return msg, nil
}

// ChatSessionStream 基于会话的流式对话
func (a *Agent) ChatSessionStream(ctx context.Context, session *Session, userMessage string, onEvent func(Event)) (string, error) {
	emit := func(e Event) {
		if onEvent != nil {
			onEvent(e)
		}
	}

	logger.Infof("[session=%s] 用户输入(stream): %s", session.ID, userMessage)
	session.AddMessage(model.Message{Role: "user", Content: userMessage})

	toolDefs := a.tools.Definitions()

	// 检查 LLM 客户端是否支持流式
	streamer, supportsStream := a.llm.(llm.StreamClient)

	for turn := 0; turn < a.maxTurns; turn++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		emit(Event{Type: EventThinking})
		history := session.GetHistory()

		// 如果支持流式，使用流式调用
		if supportsStream {
			var fullContent string
			var toolCalls []model.ToolCall
			var streamErr error

			streamErr = streamer.ChatStream(ctx, history, toolDefs, func(chunk llm.StreamChunk) {
				if chunk.Content != "" {
					fullContent += chunk.Content
					emit(Event{Type: EventStream, Content: chunk.Content})
				}
				if len(chunk.ToolCalls) > 0 {
					toolCalls = mergeToolCalls(toolCalls, chunk.ToolCalls)
				}
			})
			if streamErr != nil {
				logger.Errorf("LLM 流式调用失败: %v", streamErr)
				return "", fmt.Errorf("LLM 调用失败: %w", streamErr)
			}

			// 构建完整的 assistant 消息
			assistantMsg := model.Message{
				Role:      "assistant",
				Content:   fullContent,
				ToolCalls: toolCalls,
			}
			session.AddMessage(assistantMsg)

			if len(toolCalls) == 0 {
				logger.Infof("[session=%s] LLM 响应(stream): %s", session.ID, truncate(fullContent, 200))
				a.trimSessionByTokens(ctx, session)
				emit(Event{Type: EventStreamEnd, Content: fullContent})
				return fullContent, nil
			}

			// 处理工具调用
			for _, tc := range toolCalls {
				logger.Infof("[session=%s] 工具调用: %s args=%s", session.ID, tc.Function.Name, tc.Function.Arguments)
				emit(Event{Type: EventToolCall, Name: tc.Function.Name, Args: tc.Function.Arguments})

				result, err := a.tools.Execute(ctx, tc.Function.Name, tc.Function.Arguments)
				if err != nil {
					logger.Errorf("工具 %s 执行失败: %v", tc.Function.Name, err)
					result = fmt.Sprintf("工具执行错误: %v", err)
				}
				emit(Event{Type: EventToolResult, Name: tc.Function.Name, Result: result})
				session.AddMessage(model.Message{Role: "tool", Content: result, ToolCallID: tc.ID})
			}
			continue
		}

		// 非流式 fallback
		resp, err := a.llm.Chat(ctx, history, toolDefs)
		if err != nil {
			return "", fmt.Errorf("LLM 调用失败: %w", err)
		}
		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("LLM 返回空响应")
		}

		choice := resp.Choices[0]
		session.AddMessage(choice.Message)

		if len(choice.Message.ToolCalls) == 0 {
			a.trimSessionByTokens(ctx, session)
			emit(Event{Type: EventAssistant, Content: choice.Message.Content})
			return choice.Message.Content, nil
		}

		for _, tc := range choice.Message.ToolCalls {
			emit(Event{Type: EventToolCall, Name: tc.Function.Name, Args: tc.Function.Arguments})
			result, err := a.tools.Execute(ctx, tc.Function.Name, tc.Function.Arguments)
			if err != nil {
				result = fmt.Sprintf("工具执行错误: %v", err)
			}
			emit(Event{Type: EventToolResult, Name: tc.Function.Name, Result: result})
			session.AddMessage(model.Message{Role: "tool", Content: result, ToolCallID: tc.ID})
		}
	}

	msg := "达到最大工具调用轮次限制"
	emit(Event{Type: EventAssistant, Content: msg})
	return msg, nil
}

// trimSessionByTokens 基于 token 预算裁剪会话历史，超出时触发滚动摘要压缩
func (a *Agent) trimSessionByTokens(ctx context.Context, session *Session) {
	maxTokens := session.MaxTokens
	if maxTokens <= 0 {
		maxTokens = a.maxTokens
	}

	tokens := session.EstimateTokens()
	if tokens <= maxTokens {
		return
	}

	logger.Infof("[session=%s] token 超预算 (%d > %d)，执行滚动摘要压缩", session.ID, tokens, maxTokens)

	history := session.GetHistory()
	if len(history) <= 3 {
		return // 太少了不需要压缩
	}

	// 保留 system prompt (history[0]) 和最近的 N 条消息
	// 对中间的旧消息进行滚动摘要
	keepRecent := len(history) / 3 // 保留最近 1/3 的消息
	if keepRecent < 4 {
		keepRecent = 4
	}
	if keepRecent >= len(history)-1 {
		return
	}

	oldMessages := history[1 : len(history)-keepRecent]
	recentMessages := history[len(history)-keepRecent:]

	// 滚动摘要：旧摘要 + 中间消息 -> 增量合成新摘要，摘要永不丢失
	prevSummary := session.GetSummary()
	summary := a.summarizeMessages(ctx, prevSummary, oldMessages)
	session.SetSummary(summary)

	// 重建历史：system prompt + 摘要消息 + 最近消息（始终只保留一条摘要）
	newHistory := make([]model.Message, 0, keepRecent+2)
	newHistory = append(newHistory, history[0]) // system prompt
	newHistory = append(newHistory, model.Message{
		Role:    "system",
		Content: "[对话历史摘要] " + summary,
	})
	newHistory = append(newHistory, recentMessages...)

	session.SetHistory(newHistory)
	logger.Infof("[session=%s] 滚动摘要压缩完成: %d -> %d 条消息, 摘要长度=%d",
		session.ID, len(history), len(newHistory), len(summary))
}

// summarizeMessages 用 LLM 生成历史消息摘要（滚动式：旧摘要 + 新消息 -> 增量合成新摘要）
func (a *Agent) summarizeMessages(ctx context.Context, prevSummary string, messages []model.Message) string {
	// 构建摘要请求
	var sb strings.Builder

	if prevSummary != "" {
		sb.WriteString("【已有对话摘要】\n")
		sb.WriteString(prevSummary)
		sb.WriteString("\n\n【本次新增的对话内容】\n")
	}

	for _, msg := range messages {
		if msg.Role == "tool" {
			sb.WriteString(fmt.Sprintf("[工具返回]: %s\n", truncate(msg.Content, 100)))
		} else if msg.Role == "user" {
			sb.WriteString(fmt.Sprintf("用户: %s\n", msg.Content))
		} else if msg.Role == "assistant" {
			sb.WriteString(fmt.Sprintf("助手: %s\n", truncate(msg.Content, 200)))
		}
	}

	summarizePrompt := []model.Message{
		{Role: "system", Content: "请将对话历史压缩为简洁的摘要，保留关键信息和上下文。" +
			"如果提供了【已有对话摘要】，请在其基础上合并本次新增的对话内容，生成一份完整的更新后摘要，不要遗漏之前的关键信息。" +
			"摘要应该让AI能理解之前讨论过的内容。用中文回答，不超过300字。"},
		{Role: "user", Content: sb.String()},
	}

	resp, err := a.llm.Chat(ctx, summarizePrompt, nil)
	if err != nil {
		logger.Errorf("生成摘要失败: %v", err)
		// fallback：有旧摘要则直接沿用，否则简单截取
		if prevSummary != "" {
			return prevSummary
		}
		return truncate(sb.String(), 300)
	}
	if len(resp.Choices) > 0 {
		return resp.Choices[0].Message.Content
	}
	if prevSummary != "" {
		return prevSummary
	}
	return truncate(sb.String(), 300)
}

// mergeToolCalls 合并流式返回的 tool call 增量
// OpenAI 兼容协议中，流式 tool_calls 按 index 分片到达：
//
//	首个分片携带 id/type/name，后续分片只携带 index + arguments 片段（id/type/name 为空）。
//
// 因此必须按 index 匹配增量，而不能按 id 匹配（否则后续分片会被当成新的独立 tool call）。
func mergeToolCalls(existing []model.ToolCall, deltas []model.ToolCall) []model.ToolCall {
	for _, d := range deltas {
		found := false
		for i := range existing {
			if existing[i].Index == d.Index {
				existing[i].Function.Arguments += d.Function.Arguments
				if d.Function.Name != "" {
					existing[i].Function.Name = d.Function.Name
				}
				if d.ID != "" {
					existing[i].ID = d.ID
				}
				if d.Type != "" {
					existing[i].Type = d.Type
				}
				found = true
				break
			}
		}
		if !found {
			existing = append(existing, d)
		}
	}

	// 兜底：确保每个 tool call 都有合法的 type（OpenAI 协议要求 "function"）
	for i := range existing {
		if existing[i].Type == "" {
			existing[i].Type = "function"
		}
	}
	return existing
}

// ClearHistory 清空对话历史
func (a *Agent) ClearHistory() {
	a.history = []model.Message{
		{Role: "system", Content: a.systemPrompt},
	}
	a.notifyHistoryChanged()
}

// GetHistory 获取当前对话历史（用于持久化）
func (a *Agent) GetHistory() []model.Message {
	return a.history
}

// SetHistory 恢复对话历史（从缓存加载）
func (a *Agent) SetHistory(history []model.Message) {
	if len(history) == 0 {
		return
	}
	a.history = history
}

func (a *Agent) notifyHistoryChanged() {
	if a.onHistoryChanged != nil {
		a.onHistoryChanged()
	}
}

// trimHistory 裁剪对话历史，保留 system 消息 + 最近 maxHistory 条
func (a *Agent) trimHistory() {
	// history[0] 是 system prompt，不计入裁剪
	if len(a.history) <= a.maxHistory+1 {
		return
	}
	trimmed := make([]model.Message, 0, a.maxHistory+1)
	trimmed = append(trimmed, a.history[0]) // system prompt
	trimmed = append(trimmed, a.history[len(a.history)-a.maxHistory:]...)
	a.history = trimmed
}

// GetSystemPrompt 获取系统提示词
func (a *Agent) GetSystemPrompt() string {
	return a.systemPrompt
}

// GetMaxTokens 获取 token 预算
func (a *Agent) GetMaxTokens() int {
	return a.maxTokens
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func printHelp() {
	fmt.Print(`
可用命令:
  /help   显示帮助信息
  /clear  清空对话历史
  /quit   退出程序
  /exit   退出程序
`)
}
