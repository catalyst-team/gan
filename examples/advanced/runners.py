from catalyst_gan.core.runner import GanRunner


class FullGanRunner(GanRunner):
    # compute everything on generator train phase as well
    def _generator_train_phase(self):
        z, g_conditions = self._get_noise_and_conditions()
        d_fake_conditions = self._get_fake_data_conditions()
        d_real_conditions = self._get_real_data_conditions()

        fake_data = self.generator(z, *g_conditions)
        fake_logits = self.discriminator(
            fake_data, *d_fake_conditions  # no .detach()
        )
        real_logits = self.discriminator(
            self.state.input[self.data_input_key], *d_real_conditions
        )
        return {
            self.fake_data_output_key: fake_data,
            self.fake_logits_output_key: fake_logits,
            self.real_logits_output_key: real_logits
        }

