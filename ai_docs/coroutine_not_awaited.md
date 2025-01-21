

I run my python tests with pytest and it shows a warning:

RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited

In the first glance I didn't see problems with my code. All coroutines were awaited as far as I can see. But obviously there is some problem in tests.

How my test looks like. I deleted some info to hide business logic.

@pytest.mark.asyncio()
async def test1(
    client: httpx.AsyncClient,
    mock1: MagicMock, ...
) -> None:
    ...
    await client.post(url="url", json=jsonable_encoder(body.dict(), by_alias=True))
    mock1.func.assert_awaited_once_with(...)

As docs recommends, I've enabled tracemalloc. But it only shows this info:

.../lib/python3.11/site-packages/fastapi/routing.py:144: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    value, errors_ = field.validate(response_content, {}, loc=("response",))
  
  Object allocated at:
    File ".../lib/python3.11/site-packages/pluggy/_callers.py", line 77
      res = hook_impl.function(*args)
    File ".../lib/python3.11/site-packages/_pytest/runner.py", line 169
      item.runtest()
    File ".../lib/python3.11/site-packages/_pytest/python.py", line 1792
      self.ihook.pytest_pyfunc_call(pyfuncitem=self)
    File ".../lib/python3.11/site-packages/pluggy/_hooks.py", line 493
      return self._hookexec(self.name, self._hookimpls, kwargs, firstresult)
    File ".../lib/python3.11/site-packages/pluggy/_manager.py", line 115
      return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
    File ".../lib/python3.11/site-packages/pluggy/_callers.py", line 77
      res = hook_impl.function(*args)
    File ".../lib/python3.11/site-packages/_pytest/python.py", line 194
      result = testfunction(**testargs)
    File ".../lib/python3.11/site-packages/pytest_asyncio/plugin.py", line 532
      _loop.run_until_complete(task)
    File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/asyncio/base_events.py", line 640
      self.run_forever()
    File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/asyncio/base_events.py", line 607
      self._run_once()
    File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/asyncio/base_events.py", line 1922
      handle._run()
    File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/asyncio/events.py", line 80
      self._context.run(self._callback, *self._args)
    File ".../lib/python3.11/site-packages/starlette/middleware/base.py", line 70
      await self.app(scope, receive_or_disconnect, send_no_error)
    File ".../lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 68
      await self.app(scope, receive, sender)
    File ".../lib/python3.11/site-packages/fastapi/middleware/asyncexitstack.py", line 17
      await self.app(scope, receive, send)
    File ".../python3.11/site-packages/starlette/routing.py", line 718
      await route.handle(scope, receive, send)
    File ".../lib/python3.11/site-packages/starlette/routing.py", line 276
      await self.app(scope, receive, send)
    File ".../lib/python3.11/site-packages/starlette/routing.py", line 66
      response = await func(request)
    File ".../lib/python3.11/site-packages/fastapi/routing.py", line 291
      content = await serialize_response(
    File ".../lib/python3.11/site-packages/fastapi/routing.py", line 144
      value, errors_ = field.validate(response_content, {}, loc=("response",))

But it's fastapi package. I don't think, that the problem is there

What my possible next steps to catch where exactly not awaited coroutine created in my code? Please, recommend some tools or commands.

    python-3.xdebuggingpytestpython-asynciofastapi

Share
Improve this question
Follow
edited Nov 29, 2023 at 23:11
asked Oct 13, 2023 at 7:19
puf's user avatar
puf
46177 silver badges1616 bronze badges

    What does your tests look like? i.e. have you marked your tests with asyncio? Are you using pytest-asyncio? – 
    MatsLindh
    Commented Oct 13, 2023 at 7:23
    @MatsLindh I added test example to question. Yes, I'm using pytest-asyncio – 
    puf
    Commented Oct 13, 2023 at 7:30

Add a comment
1 Answer
Sorted by:
0

Turned out that this warning was appearing because result of function client.post was never assigned to any variable.

So I changed my test to

@pytest.mark.asyncio()
async def test1(
    client: httpx.AsyncClient,
    mock1: MagicMock, ...
) -> None:
    ...
    resp = await client.post(url="url", json=jsonable_encoder(body.dict(), by_alias=True))
    assert resp.status_code == 200
    mock1.func.assert_awaited_once_with(...)

And warning was gone.
