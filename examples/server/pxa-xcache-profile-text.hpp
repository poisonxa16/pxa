#pragma once
// PXA_XCACHE counts bootstrap: the built-in corpus the server decodes once, at startup, to measure which experts of THIS file a
// typical mix of prose, code, chat and several languages routes to (llama_pxa_xcache_bootstrap_path, pxa-xcache-bootstrap in
// server-context.cpp). Original text written for this purpose; about 3.5k tokens on a Qwen-class tokenizer.
static const char * const PXA_XC_PROFILE_TEXT = R"PXAXC(Why is the sea salty? Rain falling on land is slightly acidic because it dissolves carbon dioxide from the air. As it runs over rock it slowly dissolves minerals, and the rivers carry those dissolved ions to the ocean. Water leaves the sea again only as vapour, so the salts stay behind and accumulate over millions of years. Hydrothermal vents on the sea floor add and remove some elements as well, and the balance of all these sources and sinks keeps the average salinity close to thirty-five grams per kilogram.

The price of a good in a market is set by the interaction of supply and demand. When demand rises while supply stays fixed, buyers compete for the same goods and the price goes up, which in turn encourages producers to make more. When supply outruns demand, sellers must cut prices to clear their stock. Economists call the price at which the quantity supplied equals the quantity demanded the equilibrium price, and they study how taxes, subsidies and shocks move that point.

Photosynthesis is the process by which plants, algae and some bacteria turn light into chemical energy. In the thylakoid membranes of a chloroplast, chlorophyll absorbs photons and uses their energy to split water, releasing oxygen and passing electrons along a chain of proteins. The energy captured this way drives the synthesis of ATP and NADPH, which the Calvin cycle then uses to fix carbon dioxide into sugars.

When Johannes Gutenberg introduced movable metal type to Europe in the middle of the fifteenth century, the cost of copying a book fell dramatically. Within fifty years presses operated in hundreds of towns, pamphlets spread new ideas faster than authorities could suppress them, and literacy rose. Historians still argue about how much of the Reformation and of the scientific revolution can be traced back to that one technical change.

To bake a simple loaf of bread, mix five hundred grams of flour with three hundred and fifty grams of water, ten grams of salt and a pinch of yeast. Stir until no dry flour remains, cover the bowl and let the dough rest for twelve hours at room temperature. Fold it a few times, shape it, let it rise again for an hour, and bake it in a very hot covered pot for thirty minutes with the lid on and fifteen more with it off, until the crust is deep brown.

Entropy is a measure of how many microscopic arrangements are compatible with what we observe at the macroscopic level. A tidy room has few such arrangements, a messy one has very many, which is why messes appear by themselves and tidiness needs work. The second law of thermodynamics says that the total entropy of an isolated system never decreases, and this statistical fact is the reason heat flows from hot bodies to cold ones and not the other way round.

Philosophers have long disagreed about the foundations of ethics. A consequentialist judges an act by its results, a deontologist by whether it respects duties and rights, and a virtue ethicist by the character it expresses. Each view has famous counter-examples, and much of the modern literature consists of attempts to refine one of these positions or to show that they converge in practice.

User: Can you explain the difference between a process and a thread?
Assistant: A process is an independent program in execution with its own address space, file descriptors and resources. A thread is a unit of execution inside a process: all threads of one process share its memory, which makes communication cheap but also makes data races possible, so shared state has to be protected with locks or other synchronisation. Creating a thread is much cheaper than creating a process, and a crash in one thread usually takes down the whole process.

User: I have three apples, I eat one, then buy five more and give two away. How many do I have?
Assistant: You start with three, eating one leaves two, buying five makes seven, and giving two away leaves five apples.

User: Write a haiku about autumn rain.
Assistant: Grey clouds drift and sag, / rain taps the fallen maple leaves, / the road smells of earth.

import threading
from collections import OrderedDict


class LRUCache:
    """A small thread-safe least-recently-used cache."""

    def __init__(self, capacity: int = 128):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)


def fibonacci(n: int) -> list[int]:
    a, b = 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(size_t n) {
        for (size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] {
                for (;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
                        if (stop_ && jobs_.empty()) return;
                        job = std::move(jobs_.front());
                        jobs_.pop();
                    }
                    job();
                }
            });
        }
    }
    void submit(std::function<void()> f) {
        { std::lock_guard<std::mutex> lock(mutex_); jobs_.push(std::move(f)); }
        cv_.notify_one();
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        cv_.notify_all();
        for (auto & w : workers_) w.join();
    }
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> jobs_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

async function fetchJson(url, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise(r => setTimeout(r, 200 * attempt));
    }
  }
}

SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.id AND o.created_at >= DATE '2025-01-01'
GROUP BY c.name
HAVING SUM(o.total) > 1000
ORDER BY revenue DESC
LIMIT 20;

#!/usr/bin/env bash
set -euo pipefail
for f in logs/*.txt; do
  lines=$(wc -l < "$f")
  errors=$(grep -c "ERROR" "$f" || true)
  printf '%s: %d lines, %d errors\n' "$f" "$lines" "$errors"
done | sort -t, -k2 -nr | head -5

{
  "name": "example-service",
  "version": "1.4.2",
  "port": 8080,
  "features": { "cache": true, "metrics": ["latency", "errors"] },
  "upstreams": [ { "host": "db.internal", "weight": 3 }, { "host": "db2.internal", "weight": 1 } ]
}

| Planet  | Radius (km) | Moons |
|---------|-------------|-------|
| Mercury | 2,440       | 0     |
| Earth   | 6,371       | 1     |
| Mars    | 3,390       | 2     |
| Jupiter | 69,911      | 95    |

La inflacion es el aumento sostenido y generalizado de los precios de bienes y servicios en una economia. Cuando sube, cada unidad de moneda compra menos que antes, y por eso los bancos centrales ajustan los tipos de interes para mantenerla cerca de un objetivo.

Les mathematiques ne sont pas seulement une collection de formules : ce sont des histoires de structures et de preuves. Une demonstration elegante montre pourquoi une affirmation est vraie, et pas seulement qu'elle l'est.

Die Energiewende bezeichnet den Umbau der Energieversorgung von fossilen Brennstoffen und Kernkraft hin zu erneuerbaren Quellen wie Wind, Sonne und Biomasse. Eine Herausforderung ist die Speicherung des Stroms, wenn die Sonne nicht scheint und der Wind nicht weht.

春天来了，公园里的樱花开了，人们带着家人和朋友来散步、拍照。孩子们在草地上放风筝，老人们坐在长椅上聊天。这样的日子让人感到轻松和愉快。

日本の四季はそれぞれに美しく、春には桜、夏には花火、秋には紅葉、冬には雪景色が楽しめます。

Solve for x: 3x + 7 = 2x - 5. Subtract 2x from both sides to get x + 7 = -5, then subtract 7 to get x = -12. Check: 3(-12) + 7 = -29 and 2(-12) - 5 = -29, so the solution is correct.

A train leaves a station at 9:00 travelling at 80 km/h. A second train leaves the same station at 10:00 on the same track at 120 km/h. The first train has a head start of 80 km, and the second closes the gap at 40 km/h, so it catches up after two hours, at 12:00, 240 km from the station.

The derivative of f(x) = x^3 sin(x) is f'(x) = 3x^2 sin(x) + x^3 cos(x) by the product rule. The integral of 1/(1 + x^2) from 0 to infinity is pi/2, because the antiderivative is arctan(x).

Dear Ms. Alvarez, thank you for your message and for your patience. I have reviewed the contract you sent and have three comments: the payment schedule in section four should refer to calendar days, the liability cap in section nine seems low for a project of this size, and the termination clause should require written notice. I would be glad to discuss any of these by phone on Thursday afternoon. Kind regards, Daniel.

Breaking news from the harbour: a cargo ship that ran aground overnight has been refloated by tugboats and escorted back to deep water, officials said on Tuesday. No injuries were reported and the port has reopened to traffic, although inspectors will examine the hull before the vessel is allowed to continue to its destination.
)PXAXC";
