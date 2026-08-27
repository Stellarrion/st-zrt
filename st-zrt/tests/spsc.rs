use st_zrt::{SpscTryRecvError, SpscTrySendError, bounded_spsc};

#[test]
fn spsc_try_send_recv_preserves_order_and_reports_full() {
    let (tx, rx) = bounded_spsc::<usize>(2);

    tx.try_send(10).expect("send 10");
    tx.try_send(20).expect("send 20");
    assert_eq!(tx.try_send(30), Err(SpscTrySendError::Full(30)));

    assert_eq!(rx.try_recv().expect("recv 10"), 10);
    assert_eq!(rx.try_recv().expect("recv 20"), 20);
    assert_eq!(rx.try_recv(), Err(SpscTryRecvError::Empty));
}

#[test]
fn spsc_recv_drains_after_sender_drop() {
    let (tx, rx) = bounded_spsc::<usize>(4);
    tx.send(1).expect("send 1");
    tx.send(2).expect("send 2");
    drop(tx);

    assert_eq!(rx.recv(), Some(1));
    assert_eq!(rx.recv(), Some(2));
    assert_eq!(rx.recv(), None);
    assert_eq!(rx.try_recv(), Err(SpscTryRecvError::Closed));
}

#[test]
fn spsc_sender_observes_receiver_drop() {
    let (tx, rx) = bounded_spsc::<usize>(2);
    drop(rx);

    assert_eq!(tx.try_send(7), Err(SpscTrySendError::Closed(7)));
    assert_eq!(tx.send(8).expect_err("receiver closed").0, 8);
}

#[test]
fn spsc_rounds_capacity_and_close_prevents_sends() {
    let (tx, rx) = bounded_spsc::<usize>(3);
    assert_eq!(tx.capacity(), 4);
    assert_eq!(rx.capacity(), 4);

    tx.close();
    assert!(tx.is_closed());
    assert_eq!(tx.try_send(9), Err(SpscTrySendError::Closed(9)));
    assert_eq!(rx.recv(), None);
}

#[test]
fn spsc_threaded_roundtrip() {
    let (req_tx, req_rx) = bounded_spsc::<usize>(8);
    let (ack_tx, ack_rx) = bounded_spsc::<usize>(8);

    let worker = std::thread::spawn(move || {
        while let Some(value) = req_rx.recv() {
            ack_tx.send(value.wrapping_mul(31)).expect("ack send");
        }
    });

    for value in 0..128usize {
        req_tx.send(value).expect("request send");
        assert_eq!(ack_rx.recv(), Some(value.wrapping_mul(31)));
    }
    drop(req_tx);
    worker.join().expect("worker join");
    assert_eq!(ack_rx.recv(), None);
}
