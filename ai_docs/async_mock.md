 class unittest.mock.AsyncMock(spec=None, side_effect=None, return_value=DEFAULT, wraps=None, name=None, spec_set=None, unsafe=False, **kwargs)

    An asynchronous version of MagicMock. The AsyncMock object will behave so the object is recognized as an async function, and the result of a call is an awaitable.
    >>>

mock = AsyncMock()

asyncio.iscoroutinefunction(mock)
True

inspect.isawaitable(mock())  
True

The result of mock() is an async function which will have the outcome of side_effect or return_value after it has been awaited:

    if side_effect is a function, the async function will return the result of that function,

    if side_effect is an exception, the async function will raise the exception,

    if side_effect is an iterable, the async function will return the next value of the iterable, however, if the sequence of result is exhausted, StopAsyncIteration is raised immediately,

    if side_effect is not defined, the async function will return the value defined by return_value, hence, by default, the async function returns a new AsyncMock object.

Setting the spec of a Mock or MagicMock to an async function will result in a coroutine object being returned after calling.
>>>

async def async_func(): pass


mock = MagicMock(async_func)

mock
<MagicMock spec='function' id='...'>

mock()  
<coroutine object AsyncMockMixin._mock_call at ...>

Setting the spec of a Mock, MagicMock, or AsyncMock to a class with asynchronous and synchronous functions will automatically detect the synchronous functions and set them as MagicMock (if the parent mock is AsyncMock or MagicMock) or Mock (if the parent mock is Mock). All asynchronous functions will be AsyncMock.
>>>

class ExampleClass:

    def sync_foo():

        pass

    async def async_foo():

        pass


a_mock = AsyncMock(ExampleClass)

a_mock.sync_foo
<MagicMock name='mock.sync_foo' id='...'>

a_mock.async_foo
<AsyncMock name='mock.async_foo' id='...'>

mock = Mock(ExampleClass)

mock.sync_foo
<Mock name='mock.sync_foo' id='...'>

mock.async_foo
<AsyncMock name='mock.async_foo' id='...'>

Added in version 3.8.

assert_awaited()

    Assert that the mock was awaited at least once. Note that this is separate from the object having been called, the await keyword must be used:
    >>>

mock = AsyncMock()

async def main(coroutine_mock):

    await coroutine_mock


coroutine_mock = mock()

mock.called
True

mock.assert_awaited()
Traceback (most recent call last):
...
AssertionError: Expected mock to have been awaited.

asyncio.run(main(coroutine_mock))

    mock.assert_awaited()

assert_awaited_once()

    Assert that the mock was awaited exactly once.
    >>>

mock = AsyncMock()

async def main():

    await mock()


asyncio.run(main())

mock.assert_awaited_once()

asyncio.run(main())

    mock.assert_awaited_once()
    Traceback (most recent call last):
    ...
    AssertionError: Expected mock to have been awaited once. Awaited 2 times.

assert_awaited_with(*args, **kwargs)

    Assert that the last await was with the specified arguments.
    >>>

mock = AsyncMock()

async def main(*args, **kwargs):

    await mock(*args, **kwargs)


asyncio.run(main('foo', bar='bar'))

mock.assert_awaited_with('foo', bar='bar')

    mock.assert_awaited_with('other')
    Traceback (most recent call last):
    ...
    AssertionError: expected await not found.
    Expected: mock('other')
    Actual: mock('foo', bar='bar')

assert_awaited_once_with(*args, **kwargs)

    Assert that the mock was awaited exactly once and with the specified arguments.
    >>>

mock = AsyncMock()

async def main(*args, **kwargs):

    await mock(*args, **kwargs)


asyncio.run(main('foo', bar='bar'))

mock.assert_awaited_once_with('foo', bar='bar')

asyncio.run(main('foo', bar='bar'))

    mock.assert_awaited_once_with('foo', bar='bar')
    Traceback (most recent call last):
    ...
    AssertionError: Expected mock to have been awaited once. Awaited 2 times.

assert_any_await(*args, **kwargs)

    Assert the mock has ever been awaited with the specified arguments.
    >>>

mock = AsyncMock()

async def main(*args, **kwargs):

    await mock(*args, **kwargs)


asyncio.run(main('foo', bar='bar'))

asyncio.run(main('hello'))

mock.assert_any_await('foo', bar='bar')

    mock.assert_any_await('other')
    Traceback (most recent call last):
    ...
    AssertionError: mock('other') await not found

assert_has_awaits(calls, any_order=False)

    Assert the mock has been awaited with the specified calls. The await_args_list list is checked for the awaits.

    If any_order is false then the awaits must be sequential. There can be extra calls before or after the specified awaits.

    If any_order is true then the awaits can be in any order, but they must all appear in await_args_list.
    >>>

mock = AsyncMock()

async def main(*args, **kwargs):

    await mock(*args, **kwargs)


calls = [call("foo"), call("bar")]

mock.assert_has_awaits(calls)
Traceback (most recent call last):
...
AssertionError: Awaits not found.
Expected: [call('foo'), call('bar')]
Actual: []

asyncio.run(main('foo'))

asyncio.run(main('bar'))

    mock.assert_has_awaits(calls)

assert_not_awaited()

    Assert that the mock was never awaited.
    >>>

mock = AsyncMock()

    mock.assert_not_awaited()

reset_mock(*args, **kwargs)

    See Mock.reset_mock(). Also sets await_count to 0, await_args to None, and clears the await_args_list.

await_count

    An integer keeping track of how many times the mock object has been awaited.
    >>>

mock = AsyncMock()

async def main():

    await mock()


asyncio.run(main())

mock.await_count
1

asyncio.run(main())

    mock.await_count
    2

await_args

    This is either None (if the mock hasn’t been awaited), or the arguments that the mock was last awaited with. Functions the same as Mock.call_args.
    >>>

mock = AsyncMock()

async def main(*args):

    await mock(*args)


mock.await_args

asyncio.run(main('foo'))

mock.await_args
call('foo')

asyncio.run(main('bar'))

    mock.await_args
    call('bar')

await_args_list

    This is a list of all the awaits made to the mock object in sequence (so the length of the list is the number of times it has been awaited). Before any awaits have been made it is an empty list.
    >>>

mock = AsyncMock()

async def main(*args):

    await mock(*args)


mock.await_args_list
[]

asyncio.run(main('foo'))

mock.await_args_list
[call('foo')]

asyncio.run(main('bar'))

mock.await_args_list
[call('foo'), call('bar')]