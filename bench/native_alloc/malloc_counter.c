#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic unsigned long long g_allocs = 0;
static _Atomic unsigned long long g_frees = 0;
static _Atomic unsigned long long g_bytes = 0;

static void *(*real_malloc)(size_t) = NULL;
static void *(*real_calloc)(size_t, size_t) = NULL;
static void *(*real_realloc)(void *, size_t) = NULL;
static void (*real_free)(void *) = NULL;

static void resolve(void) {
  if (real_malloc) {
    return;
  }
  real_malloc = dlsym(RTLD_NEXT, "malloc");
  real_calloc = dlsym(RTLD_NEXT, "calloc");
  real_realloc = dlsym(RTLD_NEXT, "realloc");
  real_free = dlsym(RTLD_NEXT, "free");
  if (!real_malloc || !real_calloc || !real_realloc || !real_free) {
    fputs("zrt malloc counter: failed to resolve libc allocation symbols\n", stderr);
    abort();
  }
}

void zrt_malloc_counter_reset(void) {
  atomic_store_explicit(&g_allocs, 0, memory_order_relaxed);
  atomic_store_explicit(&g_frees, 0, memory_order_relaxed);
  atomic_store_explicit(&g_bytes, 0, memory_order_relaxed);
}

unsigned long long zrt_malloc_counter_allocs(void) {
  return atomic_load_explicit(&g_allocs, memory_order_relaxed);
}

unsigned long long zrt_malloc_counter_frees(void) {
  return atomic_load_explicit(&g_frees, memory_order_relaxed);
}

unsigned long long zrt_malloc_counter_bytes(void) {
  return atomic_load_explicit(&g_bytes, memory_order_relaxed);
}

void *malloc(size_t size) {
  resolve();
  void *p = real_malloc(size);
  if (p) {
    atomic_fetch_add_explicit(&g_allocs, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_bytes, size, memory_order_relaxed);
  }
  return p;
}

void *calloc(size_t nmemb, size_t size) {
  resolve();
  void *p = real_calloc(nmemb, size);
  if (p) {
    atomic_fetch_add_explicit(&g_allocs, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_bytes, nmemb * size, memory_order_relaxed);
  }
  return p;
}

void *realloc(void *ptr, size_t size) {
  resolve();
  void *p = real_realloc(ptr, size);
  if (p && size != 0) {
    atomic_fetch_add_explicit(&g_allocs, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_bytes, size, memory_order_relaxed);
  }
  return p;
}

void free(void *ptr) {
  resolve();
  if (ptr) {
    atomic_fetch_add_explicit(&g_frees, 1, memory_order_relaxed);
  }
  real_free(ptr);
}
