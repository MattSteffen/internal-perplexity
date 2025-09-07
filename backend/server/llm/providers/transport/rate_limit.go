package transport

import (
	"context"
	"sync"

	"golang.org/x/time/rate"
)

// Limiter provides rate limiting functionality
type Limiter struct {
	limiter *rate.Limiter
	mu      sync.RWMutex
}

// NewLimiter creates a new rate limiter with the specified requests per second and burst capacity
func NewLimiter(rps float64, burst int) *Limiter {
	return &Limiter{
		limiter: rate.NewLimiter(rate.Limit(rps), burst),
	}
}

// Wait blocks until the request can proceed
func (l *Limiter) Wait(ctx context.Context) error {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.limiter.Wait(ctx)
}

// Allow returns true if the request can proceed immediately
func (l *Limiter) Allow() bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.limiter.Allow()
}

// SetLimit updates the rate limit
func (l *Limiter) SetLimit(rps float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.limiter.SetLimit(rate.Limit(rps))
}

// SetBurst updates the burst capacity
func (l *Limiter) SetBurst(burst int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.limiter.SetBurst(burst)
}

// RateLimiter manages multiple rate limiters for different providers
type RateLimiter struct {
	limiters map[string]*Limiter
	mu       sync.RWMutex
}

// NewRateLimiter creates a new rate limiter manager
func NewRateLimiter() *RateLimiter {
	return &RateLimiter{
		limiters: make(map[string]*Limiter),
	}
}

// GetLimiter gets or creates a rate limiter for the specified provider
func (rl *RateLimiter) GetLimiter(provider string, rps float64, burst int) *Limiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if limiter, exists := rl.limiters[provider]; exists {
		return limiter
	}

	limiter := NewLimiter(rps, burst)
	rl.limiters[provider] = limiter
	return limiter
}

// WaitForProvider blocks until the request for the specified provider can proceed
func (rl *RateLimiter) WaitForProvider(ctx context.Context, provider string) error {
	rl.mu.RLock()
	limiter, exists := rl.limiters[provider]
	rl.mu.RUnlock()

	if !exists {
		// No rate limiting configured for this provider
		return nil
	}

	return limiter.Wait(ctx)
}
