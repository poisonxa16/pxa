# Beta channel (Valued Supporters)

The beta channel gives **Valued Supporters** new engine builds before they reach everyone. A beta is a preview, not the public release. The channel installs the Linux x86_64 engine build from PXA Control; the Windows and Mac downloads come from the release page. A beta installs in its own folder, next to your stable install, and **Back to stable** returns you to the stable release at any time.

Nobody has to donate to use PXA. Control and the PXQ files stay free. Donations on Ko-fi earn supporter rewards: coins for your own PXQN encodes and supporter plugins, and, for Valued Supporters, the beta.

## 1. Be a Valued Supporter

Support PXA on [Ko-fi](https://ko-fi.com/shatteredrealms1) as a Valued Supporter and connect your Discord account on Ko-fi. Ko-fi gives you the **Valued Supporter** role in the [PXA Discord](https://discord.pxanetwork.com). The beta stays yours while that membership is active.

## 2. Get your key in Discord

In the PXA Discord, type **/key**. The bot sends you your licence key by direct message; its reply in the channel is visible only to you and never shows the key. If your privacy settings block direct messages, the key comes in that private reply instead.

Keep the key private: it is tied to your Discord account. It is the same key **/encoder** gives, so you need only one.

## 3. Put the key in PXA Control

Open **Encode**, click **I have a key** (or **Get the encoder**), paste the key and press **Download Pro**. That saves the key on this computer.

## 4. Make your wallet

Open **Store**, find **PXA Coins** and press **Create my wallet**. It is free. Skip this if you already have one.

## 5. Install the beta

Stop any running model server first. Control will not restart one for you. Beta access is available while your Valued Supporter membership is active.

1. Click **Update** in the header (**Help** > **Update** opens the same window, titled **Install update**).
2. The Beta card names the release and says **Yours as a Valued Supporter.** Beta access is included and spends no coins.
3. Press **Install the beta**. A dropped download resumes the next time you press it. When this beta is already installed and nothing newer is waiting, the button reads **Reinstall check**.

Once you are on the beta channel, **Release notes** in the same window shows the notes for what you installed.

## On v3.1 or the first beta

PXA 3.1 and the first beta have no Beta card. Run **/beta** in the PXA Discord: the bot checks your Valued Supporter role and sends you personal download links for the beta by direct message. They spend no coins and work on any version. The Beta card and the **PXA Coins** panel say the same: **v3.1 / beta 1 users: run /beta in Discord.**

On 3.1 or beta 1, `pxa-update` can answer **The beta channel is for Valued Supporters; your key is not one** even when you are a Valued Supporter: beta 2 downloads through Control's wallet, which those versions do not have. Your role is fine; run **/beta** in Discord.

If you were handed a tarball, check it and unpack it into a new folder, then start Control from there:

```bash
sha256sum -c <file>.sha256
```

## Back to stable

In the same window, press **Back to stable**. If a stable build is offered, slide fully, then hold 5 seconds.

From a terminal, in the install folder:

```bash
pxa-update channel stable
pxa-update apply
```

## Report a problem

Post in **#support** on the PXA Discord. Start with the word **beta** and your version, then say what you did and what you saw. [Report a problem](REPORT-A-PROBLEM.md) shows how to attach a cleaned report from Control.

## If it does not work

- **The Beta card does not show a beta, or says it is for Valued Supporters.** Check that your Discord account shows the Valued Supporter role, that the key from **/key** is saved in **Encode**, and that you made a wallet in **Store**. Then open **Update** again. Or run **/beta** in the PXA Discord: the bot sends you personal download links.
- **You ran /key again.** That makes a new key and retires the old one. Paste the new key into **Encode** again.
- **Control says your key expired.** Run **/key** again and paste the new key.
- **pxa-update on 3.1 or beta 1 says your key is not one.** See [On v3.1 or the first beta](#on-v31-or-the-first-beta): run **/beta** in Discord.
- **The bot says the beta is for Valued Supporters.** The key service did not see the Valued Supporter role on your Discord account. Check the Ko-fi to Discord connection, then try again; nothing changes until it does.
- **Stop the running servers first.** The window says this, and the buttons wait until you do.
- **The Store is not reachable.** Wait a minute and open the window again.

The home guide is [PXA Control 4.0](CONTROL-GUIDE-4.0.md). Which file to start is [Model quick start](MODEL-QUICKSTART-4.0.md).
