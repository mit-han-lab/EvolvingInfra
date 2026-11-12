// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared test harness re-exports.

pub mod cache_aware_fixture;
pub mod mock_worker;
pub mod streaming;

/// A TCP address that refuses connections: a socket bound to a free port that
/// never listens. Keep the socket alive for the whole test; while it is bound,
/// no other listener can take the port.
#[allow(dead_code)] // Only used by some test files.
pub fn refused_addr() -> (tokio::net::TcpSocket, std::net::SocketAddr) {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = socket.local_addr().unwrap();
    (socket, addr)
}
