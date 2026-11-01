"""Shared Playwright Route / routed-page doubles for the F295-P host-side send (tests only).

WebProfiles never submits through page script: send_approved installs a one-shot page route, starts one navigation
and the host sends the approved request itself (route.fetch, max_redirects=0) or, on a bare page, passes it on with
route.fallback. A page double without route/unroute is refused ("this page cannot send a host-built request"), so
every double a submit test uses must be a RoutedPage."""


class Route(object):
    """Playwright Route double. fetch() records the host's request in page.sent and answers from page.responses
    ({url: (status, location)}, default 200) with page.response_headers ({url: {name: value}}, optional) added;
    fallback(**overrides) records what the route forced onto the request."""

    def __init__(self, url, page):
        self.request = type("Rq", (), {"url": url, "method": "GET", "headers": {"accept": "text/html"},
                                        "frame": None, "is_navigation_request": lambda self: True})()
        self.out, self.page, self.aborted, self.final = None, page, None, None

    def fallback(self, **kw):                  # a bare fallback() passes the request on to the next handler
        if kw:
            self.out = dict(kw, url=kw.get("url", self.request.url))

    def fetch(self, **kw):                     # the host's own request: recorded, answered
        self.page.sent.append(dict(kw))
        err = getattr(self.page, "fetch_error", None)    # the request went out, the answer never came (F295-L3)
        if err is not None:
            raise err
        st, loc = getattr(self.page, "responses", {}).get(kw["url"], (200, None))
        hdrs = dict(getattr(self.page, "response_headers", {}).get(kw["url"], {}))
        if loc:
            hdrs["location"] = loc
        return type("Resp", (), {"status": st, "headers": hdrs, "url": kw["url"]})()

    def fulfill(self, response=None, **kw):
        self.final = response.url if response is not None else self.request.url

    def abort(self, why=""):
        self.aborted = why


class RoutedPage(object):
    """Mixin: route / unroute and run_routes(url) -> the address the navigation ends on (raises when a route aborted).
    Uses self.routes / self.sent / self.responses, created on first use."""

    def route(self, pat, fn):
        self.__dict__.setdefault("routes", []).append(fn)

    def unroute(self, pat, fn=None):
        self.routes = [f for f in self.__dict__.get("routes", []) if fn is not None and f is not fn]

    def run_routes(self, url):
        """page_navs_before / page_navs_after: navigations the PAGE starts while the host's route is on (F295-L2);
        each outcome ("aborted" or the address it went to) is appended to page_nav_results."""
        self.__dict__.setdefault("sent", [])
        for u in self.__dict__.pop("page_navs_before", []):
            self.page_nav_results = getattr(self, "page_nav_results", []) + [self._page_nav(u)]
        out = self._run(url)
        for u in self.__dict__.pop("page_navs_after", []):
            self.page_nav_results = getattr(self, "page_nav_results", []) + [self._page_nav(u)]
        return out

    def _page_nav(self, u):
        try:
            return self._run(u)
        except RuntimeError:
            return "aborted"

    def _run(self, url):
        for fn in reversed(self.__dict__.get("routes", [])):   # Playwright: the last registered route runs first
            r = Route(url, self)
            fn(r)
            if r.aborted is not None:
                raise RuntimeError("net::ERR_BLOCKED_BY_CLIENT")
            if r.out is not None:
                self.sent.append(r.out)
                return r.out["url"]
            if r.final is not None:
                return url
        return url
