

# PUT TRAIN CODE HERE (TRAIN)

def train(input_tensor, target_tensor, encoder, decoder, optimizer,
          criterion, args):

    # This roughly mirrors the provided translate.translate...
    optimizer.zero_grad()

    # Toggle training mode so dropout is enabled
    encoder.train()
    decoder.train()

    # Run input_tensor word-by-word through encoder
    encoder_outputs, hidden = encoder(input_tensor)

    loss = 0.0

    decoder_output = None

    if args.teacher_forcing:
        decoder_output, hidden, _ = decoder(encoder_outputs,
                                            target_tensor, hidden)
    else:
        decoder_output, hidden, _ = decoder(encoder_outputs, None, hidden)


    loss += criterion(decoder_output, target_tensor)

    loss.backward()
    optimizer.step()

    return loss.item()
