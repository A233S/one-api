package controller

import (
	"fmt"
	"net/http"
	"one-api/common"
	"one-api/model"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

type Message struct {
	Role    string  `json:"role"`
	Content string  `json:"content"`
	Name    *string `json:"name,omitempty"`
}

const (
	RelayModeUnknown = iota
	RelayModeChatCompletions
	RelayModeCompletions
	RelayModeEmbeddings
	RelayModeModerations
	RelayModeImagesGenerations
	RelayModeEdits
	RelayModeAudio
)

// https://platform.openai.com/docs/api-reference/chat

type GeneralOpenAIRequest struct {
	Model       string    `json:"model,omitempty"`
	Messages    []Message `json:"messages,omitempty"`
	Prompt      any       `json:"prompt,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	TopP        float64   `json:"top_p,omitempty"`
	N           int       `json:"n,omitempty"`
	Input       any       `json:"input,omitempty"`
	Instruction string    `json:"instruction,omitempty"`
	Size        string    `json:"size,omitempty"`
	Functions   any       `json:"functions,omitempty"`
}

type ChatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	MaxTokens int       `json:"max_tokens"`
}

type TextRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Prompt    string    `json:"prompt"`
	MaxTokens int       `json:"max_tokens"`
	//Stream   bool      `json:"stream"`
}

type ImageRequest struct {
	Prompt string `json:"prompt"`
	N      int    `json:"n"`
	Size   string `json:"size"`
}

type AudioResponse struct {
	Text string `json:"text,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    any    `json:"code"`
}

type OpenAIErrorWithStatusCode struct {
	OpenAIError
	StatusCode int `json:"status_code"`
}

type TextResponse struct {
	Choices []OpenAITextResponseChoice `json:"choices"`
	Usage   `json:"usage"`
	Error   OpenAIError `json:"error"`
}

type OpenAITextResponseChoice struct {
	Index        int `json:"index"`
	Message      `json:"message"`
	FinishReason string `json:"finish_reason"`
}

type OpenAITextResponse struct {
	Id      string                     `json:"id"`
	Object  string                     `json:"object"`
	Created int64                      `json:"created"`
	Choices []OpenAITextResponseChoice `json:"choices"`
	Usage   `json:"usage"`
}

type OpenAIEmbeddingResponseItem struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type OpenAIEmbeddingResponse struct {
	Object string                        `json:"object"`
	Data   []OpenAIEmbeddingResponseItem `json:"data"`
	Model  string                        `json:"model"`
	Usage  `json:"usage"`
}

type ImageResponse struct {
	Created int `json:"created"`
	Data    []struct {
		Url string `json:"url"`
	}
}

type ChatCompletionsStreamResponseChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type ChatCompletionsStreamResponse struct {
	Id      string                                `json:"id"`
	Object  string                                `json:"object"`
	Created int64                                 `json:"created"`
	Model   string                                `json:"model"`
	Choices []ChatCompletionsStreamResponseChoice `json:"choices"`
}

type CompletionsStreamResponse struct {
	Choices []struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func Relay(c *gin.Context) {
	relayMode := RelayModeUnknown
	if strings.HasPrefix(c.Request.URL.Path, "/v1/chat/completions") {
		relayMode = RelayModeChatCompletions
	} else if strings.HasPrefix(c.Request.URL.Path, "/v1/completions") {
		relayMode = RelayModeCompletions
	} else if strings.HasPrefix(c.Request.URL.Path, "/v1/embeddings") {
		relayMode = RelayModeEmbeddings
	} else if strings.HasSuffix(c.Request.URL.Path, "embeddings") {
		relayMode = RelayModeEmbeddings
	} else if strings.HasPrefix(c.Request.URL.Path, "/v1/moderations") {
		relayMode = RelayModeModerations
	} else if strings.HasPrefix(c.Request.URL.Path, "/v1/images/generations") {
		relayMode = RelayModeImagesGenerations
	} else if strings.HasPrefix(c.Request.URL.Path, "/v1/edits") {
		relayMode = RelayModeEdits
	} else if strings.HasPrefix(c.Request.URL.Path, "/v1/audio") {
		relayMode = RelayModeAudio
	}
	var err *OpenAIErrorWithStatusCode
	switch relayMode {
	case RelayModeImagesGenerations:
		err = relayImageHelper(c, relayMode)
	case RelayModeAudio:
		err = relayAudioHelper(c, relayMode)
	default:
		err = relayTextHelper(c, relayMode)
	}
	if err != nil {
		retryTimesStr := c.Query("retry")
		retryTimes, _ := strconv.Atoi(retryTimesStr)
		if retryTimesStr == "" {
			retryTimes = common.RetryTimes
		}
		if retryTimes > 0 {
			c.Redirect(http.StatusTemporaryRedirect, fmt.Sprintf("%s?retry=%d", c.Request.URL.Path, retryTimes-1))
		} else {
			if err.StatusCode == http.StatusTooManyRequests {
				err.OpenAIError.Message = "当前分组上游负载已饱和，请稍后再试"
			}
			c.JSON(err.StatusCode, gin.H{
				"error": err.OpenAIError,
			})
		}
		channelId := c.GetInt("channel_id")
		common.SysError(fmt.Sprintf("relay error (channel #%d): %s", channelId, err.Message))
		// https://platform.openai.com/docs/guides/error-codes/api-errors
		if shouldDisableChannel(&err.OpenAIError, err.StatusCode) {
			channelId := c.GetInt("channel_id")
			channelName := c.GetString("channel_name")
			disableChannel(channelId, channelName, err.Message)
		}
		if shouldTemporarilyDisableChannel(c, &err.OpenAIError, err.StatusCode) {
			channelId := c.GetInt("channel_id")
			channelName := c.GetString("channel_name")
			retryInterval := c.GetInt("retryInterval")
			if retryInterval == 0 {
				retryInterval = 300
			}
			disableChannel(channelId, channelName, err.Message)
			channel, err2 := model.GetChannelById(channelId, true)
			if err2 != nil {
				common.SysError(fmt.Sprintf("failed to get channel: %s", err2.Error()))
				return
			}
			// 启动goroutine执行定时任务,定时轮训需要启动的任务
// ... [之前的代码]

go func() {
    tryCount := 0 // 初始化尝试计数器
    waitTime := 15 // 初始等待时间为 15 分钟
    for {
        if tryCount >= 20 { // 检查尝试次数是否达到限制
            fmt.Println("Auto channel test: %s 已超过最大重试次数，停止尝试")
            return
        }

        // 根据等差数列调整等待时间
        time.Sleep(time.Duration(waitTime) * time.Minute)
        // 模拟修改数据状态的过程
        testRequest := buildTestRequest()
        err3, aiError := testChannel(channel, *testRequest)
        if err3 != nil {
            common.SysError(fmt.Sprintf("failed to test channel: %s", err3.Error()))
            tryCount++ // 增加尝试计数
            waitTime += 60 // 按照等差数列增加等待时间
            continue
        }
        if aiError != nil {
            common.SysError(fmt.Sprintf("failed to test channel: %s", aiError.Message))
            tryCount++ // 增加尝试计数
            waitTime += 60 // 按照等差数列增加等待时间
            continue
        }
        // 自动启用通道
        channel.Status = 1
        err := channel.Update()
        if err != nil {
            common.SysError(fmt.Sprintf("failed to update channel: %s", err.Error()))
            return
        }
        // 修改数据状态成功
        fmt.Println("数据状态修改成功")
        // 关闭当前goroutine
        return
    }
}()


// ... [后续代码]


		}
	}
}

func RelayNotImplemented(c *gin.Context) {
	err := OpenAIError{
		Message: "API not implemented",
		Type:    "one_api_error",
		Param:   "",
		Code:    "api_not_implemented",
	}
	c.JSON(http.StatusNotImplemented, gin.H{
		"error": err,
	})
}

func RelayNotFound(c *gin.Context) {
	err := OpenAIError{
		Message: fmt.Sprintf("Invalid URL (%s %s)", c.Request.Method, c.Request.URL.Path),
		Type:    "invalid_request_error",
		Param:   "",
		Code:    "",
	}
	c.JSON(http.StatusNotFound, gin.H{
		"error": err,
	})
}
