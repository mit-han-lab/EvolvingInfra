"""Real tokenizer-owned HTTP pool against an independent Uvicorn process."""
import asyncio
import multiprocessing
import socket
import time

from evolving_infra.router.frontend.protocol import NodeConfig
from evolving_infra.router.frontend.server import TokenizerService


def upstream(pipe):
    import uvicorn

    async def app(scope, receive, send):
        while True:
            event = await receive()
            if not event.get('more_body', False):
                break
        pipe.send(scope['path'])
        if scope['path'] == '/warm':
            await asyncio.sleep(.15)
        await send(dict(type='http.response.start', status=200,
                        headers=[(b'content-length', b'2')]))
        await send(dict(type='http.response.body', body=b'ok'))

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    pipe.send(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, lifespan='off', loop='asyncio',
        http='h11', timeout_keep_alive=5, access_log=False, log_level='warning'))
    server.run(sockets=[sock])


def test_owned_tokenizer_client_survives_response_delivered_during_loop_pause():
    async def scenario():
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=upstream, args=(child,))
        process.start()
        child.close()
        task = None
        try:
            assert parent.poll(10), 'Upstream did not bind'
            url = f'http://127.0.0.1:{parent.recv()}'
            await asyncio.sleep(.3)
            config = NodeConfig(node_id='test', config_id='test', model_path='/model',
                tokenize_endpoint=url + '/warm', targets={url + '/probe': 'test'},
                context_length=4096, page_size=4)
            service = TokenizerService(config, hasher=object())
            async with service.lifespan(service.app):
                async def requests():
                    async with service.client.post(url + '/warm', data=b'one') as response:
                        await response.read()
                    # Same coroutine: do not yield an extra task-join turn before
                    # reuse, which would let a queued peer EOF hide the bug.
                    async with service.client.post(url + '/probe', data=b'two') as response:
                        assert await response.read() == b'ok'
                task = asyncio.create_task(requests())
                assert await asyncio.wait_for(asyncio.to_thread(parent.recv), 5) == '/warm'
                time.sleep(6.5)  # Realistic long GC/stall; upstream runs separately.
                await asyncio.wait_for(task, 5)
                assert parent.poll(1) and parent.recv() == '/probe'
                assert not parent.poll(.05), 'Unexpected generation retry'
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            parent.close()
    asyncio.run(scenario())
